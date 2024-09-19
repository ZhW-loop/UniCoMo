from typing import Dict
import pickle, glob, json, sys
import torch
from tvm.ir import load_json
from tvm.meta_schedule.utils import shash2hex
import numpy as np
import argparse
from tlp_train import SegmentDataLoader, AttentionModule
from common import TensorizeSet_dir
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_test_tasks(measured_records, hold_out_models, task_info_path):
    hold_out_workload_keys:Dict[str, Dict]  = {}
    hold_out_models_paths = []
    task_paths = sorted(glob.glob(os.path.join(task_info_path, "*.json")))[
        0 :
    ]
    
    for hold_out_model in hold_out_models:
        for task_path in task_paths:
            if hold_out_model in task_path:
                hold_out_models_paths.append(task_path)
    
    for hold_out_models_path in hold_out_models_paths:
        hold_out_workload_keys[hold_out_models_path] = {}
        with open(hold_out_models_path, "rb") as file:
            tasks = file.readlines()
            for task_str in tasks:
                task_name, task_mod, _, weight = json.loads(task_str)
                task_mod = load_json(json.dumps(task_mod))
                hold_out_workload_keys[hold_out_models_path][shash2hex(task_mod)] = weight
                
    return hold_out_workload_keys



top_ks = [1, 5, 10, 20]

def pred_a_dataset(datas, task_pred_dict, model):

    datas_new = []
    for data_idx, data in enumerate([datas]):
        workloadkey, line_vecs = data
        datas_new.extend(line_vecs)

    # if isinstance(model, BertModule):
    #     test_loader = BertSegmentDataLoader(datas_new, 512, False)
    # elif isinstance(model, GPTModule):
    #     test_loader = GPTSegmentDataLoader(datas_new, 512, False)
    # else:
        test_loader = SegmentDataLoader(datas_new, 4000, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_all = []
    labels_all = []

    for batch_datas_steps, batch_labels in test_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        preds = model(batch_datas_steps)
        # if isinstance(preds, list) and len(preds) > 1:
        #     preds = preds[0]
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    task_pred_dict[workloadkey] = (preds_all.detach().cpu().numpy(
    ), test_loader.min_latency.min().numpy(), labels_all.numpy())


def eval_model(model_file):

    with open(model_file, 'rb') as f:
        # model = pickle.load(f).module.to(device)
        model = pickle.load(f).to(device)
    model.eval()
    task_pred_dict = {}

    pred_a_dataset_dict = {}
    for data_idx, data in enumerate(test_datasets):
        workloadkey, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

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
    files = get_test_tasks(measured_path, hold_out_models, task_info_path)
    
    top_1_total = []
    top_5_total = []
    top_10_total = []
    top_20_total = []
    best_latency_total_list = []
    best_latency_total = 0
    top1_total = 0
    top5_total = 0
    top10_total = 0
    top20_total = 0
    for path, file in files.items():
        latencies = [0] * len(top_ks)
        best_latency = 0

        for workload_key, weight in file.items():
            if workload_key not in pred_a_dataset_dict:
                print('task.workload_key not in pred_a_dataset_dict skipping ...')
                continue
            print(workload_key)
            pred_a_dataset(
                pred_a_dataset_dict[workload_key], task_pred_dict, model)
            preds, min_latency, labels = task_pred_dict[workload_key]

            real_values = labels[np.argsort(-preds)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_latency * weight
            print(f"pred: {real_latency[:5]},\tpred_rank: {np.argsort(real_latency)[:5]},\tmin: {min_latency},\tweight: {weight}" )
        if best_latency == 0 and latencies == [0] * len(top_ks): continue
        top_1_total.append(best_latency/latencies[0])
        print(f"top 1 score: {best_latency/latencies[0]}")
        top_5_total.append(best_latency / latencies[1])
        print(f"top 5 score: {best_latency / latencies[1]}")

        best_latency_total_list.append(best_latency)
        best_latency_total += best_latency
        top1_total += latencies[0]
        top5_total += latencies[1]
        top10_total += latencies[2]
        top20_total += latencies[3]


    print(f"average top 1 score is {best_latency_total / top1_total}")
    top_1_total.append(best_latency_total / top1_total)
    print(f"average top 5 score is {best_latency_total / top5_total}")
    top_5_total.append(best_latency_total / top5_total)
    print(f"average top 10 score is {best_latency_total / top10_total}")
    top_10_total.append(best_latency_total / top10_total)
    print(f"average top 20 score is {best_latency_total / top20_total}")
    top_20_total.append(best_latency_total / top20_total)



if __name__ == "__main__":
    # measured_records = '/home/zhwang/TensorizeSet/tlp_dataset/measured_records'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    parser.add_argument("--model_id", type=str, default='37')
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--test_dataset_name", type=str, default='')
    parser.add_argument("--load_name", type=str, default='') 
    parser.add_argument("--platform", type=str, default='llvm')  # or cuda
    args = parser.parse_args()
    print(args)

    intrin_name = args.intrin_name
    model_id = args.model_id
    measured_path = os.path.join(TensorizeSet_dir, f'{intrin_name}_dataset/measured_records') 
    dataset_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/tlp_dataset')
    train_save_name = f'tlp_{intrin_name}_train_and_val.pkl.DataLoader'
    eval_save_name = f'tlp_{intrin_name}_test.pkl'
    
    model_save_path = os.path.join(TensorizeSet_dir, f'tlp_records/tlp_models/{intrin_name}') 
    
    task_info_path = os.path.join(TensorizeSet_dir, 'task_info')
    device = args.cuda

    if args.test_dataset_name == '':
        args.test_dataset_name = os.path.join(dataset_save_path, eval_save_name)
    if args.load_name == '':
        args.load_name = f"tlp_model_{model_id}.pkl"
    
    with open(args.test_dataset_name, 'rb') as f:
        test_datasets = pickle.load(f)

    load_name = os.path.join(model_save_path, args.load_name)
    eval_model(load_name)