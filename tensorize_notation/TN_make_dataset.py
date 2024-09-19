import sys
sys.path.append("TensorizeSet_app")
from TensorizeSet import common
from TensorizeSet.pkl2DataLoader import SegmentDataLoader, load_datas

import os, glob, multiprocessing, numpy as np, pickle, random, json
from typing import List

import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import shash2hex

from tvm.ir import load_json
from tvm.meta_schedule import FeatureExtractor
import tvm.tir.tensor_intrin

import argparse
from common import TensorizeSet_dir

def get_hold_out_tasks(measured_records, hold_out_models, task_info_path):
    print("Entering get_hold_out_tasks ...")
    hold_out_workload_keys = {}
    hold_out_models_paths = []
    task_paths = sorted(glob.glob(os.path.join(task_info_path, "*.json")))[
        0 :
    ]
    
    for hold_out_model in hold_out_models:
        for task_path in task_paths:
            if hold_out_model in task_path:
                hold_out_models_paths.append(task_path)
    
    for hold_out_models_path in hold_out_models_paths:
        with open(hold_out_models_path, "rb") as file:
            tasks = file.readlines()
            for task_str in tasks:
                task_name, task_mod, _, weight = json.loads(task_str)
                task_mod = load_json(json.dumps(task_mod))
                hold_out_workload_keys[shash2hex(task_mod)] = weight
    assert len(hold_out_workload_keys) > 0
    print("get_hold_out_tasks Done ...")
    print(hold_out_workload_keys)
    return hold_out_workload_keys

def handle_file(idx, workload_json_file, candidates_json_file, 
                embedding_utils: FeatureExtractor, target = common.TENSORCORE_TARGET):
    print(idx, end=" ", flush=True)
    database = ms.database.JSONDatabase(
                    path_workload=workload_json_file,
                    path_tuning_record=candidates_json_file,
                )
    tuning_records = database.get_all_tuning_records()
    tuning_records = list(tuning_records)
    if len(tuning_records) == 0:
        return None
    workload_key = shash2hex(tuning_records[0].workload.mod)
    line_vecs = []     # all lines in one file, e.g. all schedule traces in one file
    min_cost = 1e6
    candidates = []
    for tuning_record in tuning_records:
        candidates.append(tuning_record.as_measure_candidate())
    context = ms.TuneContext(
            mod=tuning_records[0].workload.mod,
            target=tvm.target.Target(target),
        )
    scores = embedding_utils.extract_from(context, candidates)
    xs=[ x.numpy().astype("double").tolist() for x in scores ]
    
    assert len(xs) == len(tuning_records)
    
    for i, tuning_record in enumerate(tuning_records):
        costs = [x.value for x in tuning_record.run_secs]
        cost = np.mean(costs)
        min_cost = min(min_cost, cost)
        line_vecs.append((xs[i], cost))

    line_vecs_new = []
    for line_vec in line_vecs:
        x, cost = line_vec
        score = min_cost / cost
        line_vecs_new.append((x, score, min_cost))
    line_vecs = line_vecs_new
    del database, tuning_records
    return (workload_key, line_vecs)
        

def handle_file_pkl(idx, workload_json_file, candidates_json_file, 
                embedding_utils: FeatureExtractor, target = common.TENSORCORE_TARGET):
    print(idx, end=" ", flush=True)
    if os.path.exists(workload_json_file.replace('_workload.json', '_TN.pkl')):
        print(f"{workload_json_file.replace('_workload.json', '_TN.pkl')} already exists, skipping ...")
        return None
    database = ms.database.JSONDatabase(
                    path_workload=workload_json_file,
                    path_tuning_record=candidates_json_file,
                )
    tuning_records = database.get_all_tuning_records()
    tuning_records = list(tuning_records)
    if len(tuning_records) == 0:
        return None
    workload_key = shash2hex(tuning_records[0].workload.mod)
    line_vecs = []     # all lines in one file, e.g. all schedule traces in one file
    min_cost = 1e6
    candidates = []
    for tuning_record in tuning_records:
        candidates.append(tuning_record.as_measure_candidate())
    context = ms.TuneContext(
            mod=tuning_records[0].workload.mod,
            target=tvm.target.Target(target),
        )
    scores = embedding_utils.extract_from(context, candidates)
    xs=[ x.numpy().astype("double").tolist() for x in scores ]
    
    assert len(xs) == len(tuning_records)
    
    for i, tuning_record in enumerate(tuning_records):
        costs = [x.value for x in tuning_record.run_secs]
        cost = np.mean(costs)
        min_cost = min(min_cost, cost)
        line_vecs.append((xs[i], cost))

    line_vecs_new = []
    for line_vec in line_vecs:
        x, cost = line_vec
        score = min_cost / cost
        line_vecs_new.append((x, score, min_cost))
    line_vecs = line_vecs_new
    
    save_pkl_path = workload_json_file.replace('_workload.json', '_TN.pkl')
    with open(save_pkl_path, 'wb') as f:
        pickle.dump((workload_key, line_vecs), f)
    
    del workload_key, line_vecs, database, tuning_records


def make_all_dataset(measured_records: str, 
                     embedding_utils: FeatureExtractor,
                     hold_out_models: List[str], 
                     files_cnt=-1):
    workload_json_files = []
    candidates_json_files = []
    
    # fill in workload and candidates
    model_dirs = sorted(glob.glob(os.path.join(measured_records, "*")))[0:]
    for model_dir in model_dirs:
        wjf = glob.glob(os.path.join(model_dir, "*_workload.json"))
        workload_json_files.extend(wjf)
    if files_cnt > 0:
        workload_json_files = random.sample(workload_json_files, files_cnt)
    else:
        files_cnt = len(workload_json_files)
    
    # add test e.g. hold_out_models
    for hold_out_model in hold_out_models:
        hold_out_workloads = glob.glob(os.path.join(measured_records, 
                                                   hold_out_model, "*_workload.json"))
        for hold_out_workload in hold_out_workloads:
            if hold_out_workload not in workload_json_files:
                workload_json_files.append(hold_out_workload)
    
    workload_json_files = sorted(workload_json_files)
    for workload_json_file in workload_json_files:
        candidates_json_files.append(workload_json_file.replace("_workload.json", 
                                                                "_candidates.json"))
    
    multiprocessing_pool = multiprocessing.Pool(processes=32)
    que_res_list = [] 
    for idx, (workload_json_file, candidates_json_file) in \
        enumerate(zip(workload_json_files, candidates_json_files)):
                que_res_list.append(multiprocessing_pool.apply_async(
                    handle_file, 
                    args=(idx, workload_json_file, candidates_json_file, embedding_utils)))
    
    # for idx, (workload_json_file, candidates_json_file) in \
    #     enumerate(zip(workload_json_files, candidates_json_files)):
    #             que_res_list.append(handle_file(idx, workload_json_file, candidates_json_file,
    #                                             embedding_utils))
    
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    file_vecs = []
    task_set = {}
    repeat = 0
    empty = 0
    for que_res in que_res_list:
        file_vec = que_res.get()
        if file_vec == None:
            empty += 1
            continue
        if file_vec[0] in task_set:
            repeat += 1
            continue
        file_vecs.append(que_res.get())
        task_set[file_vec[0]] = 1
    print(f"make_all_dataset, repeat: {repeat}.")
    print(f"make_all_dataset, empty: {empty}.")
    return file_vecs

def make_all_dataset_pkl(measured_records: str, 
                     embedding_utils: FeatureExtractor,
                     hold_out_models: List[str], 
                     files_cnt=-1):
    workload_json_files = []
    candidates_json_files = []
    
    # fill in workload and candidates
    model_dirs = sorted(glob.glob(os.path.join(measured_records, "*")))[0:]
    for model_dir in model_dirs:
        wjf = glob.glob(os.path.join(model_dir, "*_workload.json"))
        workload_json_files.extend(wjf)
    if files_cnt > 0:
        workload_json_files = random.sample(workload_json_files, files_cnt)
    else:
        files_cnt = len(workload_json_files)
    
    # add test e.g. hold_out_models
    for hold_out_model in hold_out_models:
        hold_out_workloads = glob.glob(os.path.join(measured_records, 
                                                   hold_out_model, "*_workload.json"))
        for hold_out_workload in hold_out_workloads:
            if hold_out_workload not in workload_json_files:
                workload_json_files.append(hold_out_workload)
    
    workload_json_files = sorted(workload_json_files)
    for workload_json_file in workload_json_files:
        candidates_json_files.append(workload_json_file.replace("_workload.json", 
                                                                "_candidates.json"))
    
    multiprocessing_pool = multiprocessing.Pool(processes=16)
    que_res_list = [] 
    for idx, (workload_json_file, candidates_json_file) in \
        enumerate(zip(workload_json_files, candidates_json_files)):
                multiprocessing_pool.apply_async(
                    handle_file_pkl, 
                    args=(idx, workload_json_file, candidates_json_file, embedding_utils))
    
    # for idx, (workload_json_file, candidates_json_file) in \
    #     enumerate(zip(workload_json_files, candidates_json_files)):
    #             que_res_list.append(handle_file(idx, workload_json_file, candidates_json_file,
    #                                             embedding_utils))
    
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    
    # collect all pkl
    pkl_files = []
    for model_dir in model_dirs:
        pjf = glob.glob(os.path.join(model_dir, "*_TN.pkl"))
        pkl_files.extend(pjf)
    
    for pf in pkl_files:
        with open(pf, 'rb') as f:
            workload_key, line_vecs = pickle.load(f)
        que_res_list.append((workload_key, line_vecs))
        del workload_key, line_vecs
    
    file_vecs = []
    task_set = {}
    repeat = 0
    empty = 0
    for que_res in que_res_list:
        file_vec = que_res.get()
        if file_vec == None:
            empty += 1
            continue
        if file_vec[0] in task_set:
            repeat += 1
            continue
        file_vecs.append(file_vec)
        task_set[file_vec[0]] = 1
    print(f"make_all_dataset, repeat: {repeat}.")
    print(f"make_all_dataset, empty: {empty}.")
    return file_vecs


def split_dataset(file_vecs, hold_out_tasks_set, measured_records, files_cnt, dataset_save_path, train_save_name, eval_save_name):
    # comp files_cnt
    workload_json_files = []
    model_dirs = sorted(glob.glob(os.path.join(measured_records, "*")))[0:]
    for model_dir in model_dirs:
        wjf = glob.glob(os.path.join(model_dir, "*_workload.json"))
        workload_json_files.extend(wjf)
    if files_cnt < 0: 
        files_cnt = len(workload_json_files)
    
    train_and_val_dataset = []
    test_dataset = []
    for file_vec_idx, file_vec in enumerate(file_vecs):
        # file_vec == line_vecs
        workload_key, line_vecs = file_vec
        # print(f"{file_vec_idx}, {len(line_vecs)}")
        
        if workload_key in hold_out_tasks_set:
            test_dataset.append(file_vec)
        else:
            train_and_val_dataset.append(file_vec)
    train_and_val_dataset_new = []
    for data_idx, data in enumerate(train_and_val_dataset):
        workload_key, line_vecs = data
        train_and_val_dataset_new.extend(line_vecs)
    
    
    with open(os.path.join(dataset_save_path, eval_save_name), 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(os.path.join(dataset_save_path, train_save_name), 'wb') as f:
        pickle.dump(train_and_val_dataset_new, f)
    # make DataLoader
    dataloader_path = str(os.path.join(dataset_save_path, train_save_name)) + ".DataLoader"
    train_dataloader, val_dataloader = load_datas(train_and_val_dataset_new)
    with open(dataloader_path, 'wb') as f:
        pickle.dump((train_dataloader, val_dataloader), f)
        

def get_dataset(measured_records, train_and_val, test,):
    # train_and_val_dataset[0] = (schedule_vecs, score, min_cost)
    # train_and_val_dataset[0][0] = schedule_vecs
    # train_and_val_dataset[0][0][0] = vec
    train_and_val_dataset = \
        pickle.load(open(os.path.join(measured_records, train_and_val), 'rb'))
    # test_dataset[0] = (workload_key, line_vecs)
    # test_dataset[0][1] = line_vecs
    # test_dataset[0][1][0] = (schedule_vecs, score, min_cost)
    # test_dataset[0][1][0][0] = schedule_vecs
    # test_dataset[0][1][0][0][0] = vec
    test_dataset = \
        pickle.load(open(os.path.join(measured_records, test), 'rb'))
    return train_and_val_dataset, test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    args = parser.parse_args()
    intrin_name = args.intrin_name
    
    measured_path = os.path.join(TensorizeSet_dir, f'{intrin_name}_dataset/measured_records') 
    dataset_save_path = os.path.join(TensorizeSet_dir, 'TN_records/TN_dataset')
    train_save_name = f'TN_{intrin_name}_train_and_val.pkl'
    eval_save_name = f'TN_{intrin_name}_test.pkl'
    
    table_save_path = os.path.join(TensorizeSet_dir, 'TN_records/TN_table')
    table_save_name = f'TN_{intrin_name}_embedding_table.pkl'
    
    task_info_path = os.path.join(TensorizeSet_dir, 'task_info')
    files_cnt = -1
    
    if "tensorcore" in intrin_name:
        hold_out_models = [
                        # 'bert_large-None-1,128-tensorcore-float16',
                        'bert_base-None-1,128-tensorcore-float16',
                        'bert_tiny-None-1,128-tensorcore-float16',
                        'resnet_50-NHWC-1,3,224,224-tensorcore-float16',
                        # 'resnet_18-NHWC-1,3,224,224-tensorcore-float16',
                        'mobilenet_v2-NHWC-1,3,224,224-tensorcore-float16',
                        # 'densenet_121-NHWC-1,3,224,224-tensorcore-float16',
                        # 'inception_v3-NHWC-1,3,299,299-tensorcore-float16',
                        ] # model name
    if "avx512" in intrin_name or "vnni" in intrin_name :
        hold_out_models = [
                        # 'bert_large-None-1,128-avx512-int8',
                        'bert_base-None-1,128-avx512-int8',
                        'bert_tiny-None-1,128-avx512-int8',
                        'resnet_50-NCHW-1,3,224,224-avx512-int8',
                        # 'resnet_18-NCHW-1,3,224,224-avx512-int8',
                        'mobilenet_v2-NCHW-1,3,224,224-avx512-int8',
                        # 'densenet_121-NCHW-1,3,224,224-avx512-int8',
                        # 'inception_v3-NCHW-1,3,299,299-avx512-int8',
                        ] # model name
    if "neon" in intrin_name or "sdot" in intrin_name:
        hold_out_models = [
                        # 'bert_large-None-1,128-neon-int8',
                        'bert_base-None-1,128-neon-int8',
                        'bert_tiny-None-1,128-neon-int8',
                        'resnet_50-NCHW-1,3,224,224-neon-int8',
                        # 'resnet_18-NCHW-1,3,224,224-neon-int8',
                        'mobilenet_v2-NCHW-1,3,224,224-neon-int8',
                        # 'densenet_121-NCHW-1,3,224,224-neon-int8',
                        # 'inception_v3-NCHW-1,3,299,299-neon-int8',                       
                        ] # model name
    
    hold_out_tasks_set = get_hold_out_tasks(measured_path, hold_out_models, task_info_path)
    
    embedding_utils = FeatureExtractor.create('per-block-feature')
    
    # file_vecs = make_all_dataset(measured_records, embedding_utils, hold_out_models, files_cnt)
    file_vecs = make_all_dataset(measured_path, embedding_utils, hold_out_models, files_cnt)
    split_dataset(file_vecs, hold_out_tasks_set, measured_path, files_cnt, dataset_save_path, train_save_name, eval_save_name)
    print('make dataset TN Done.')
    
    # train_and_val_dataset, test_dataset = \
    #     get_dataset(measured_records, 
    #                 f'TN_{files_cnt}_train_and_val.pkl',
    #                 f'TN_{files_cnt}_test.pkl',)
    # print(train_and_val_dataset[0][0][20])
    # print(test_dataset[0][1][0][0][20])
    
    

# NOTE
# 204 only resnet
# 135 only bert
# 339 bert and resnet
# 770 bert, resnet, densnet
# 1116 bert, resnet, densnet, mobilenet_v2, inception_v3, resnet3d_18