from typing import Dict,List
import glob
import os, sys, argparse
import multiprocessing
import pickle

import tvm
import tvm.tir.tensor_intrin
from tvm import meta_schedule as ms
from sch_trace_process import is_float, Sch2Listdict
from common import TensorizeSet_dir

def handle_file(idx, workload_json_file, candidates_json_file):
    print(idx, end=" ", flush=True)
    local_str_embedding: Dict[str, float] = {}
    local_kind_embedding: Dict[str, List] = {}
    local_max_embedding_len: int = -1
    local_max_seq_len: int = -1
    try:
        database = ms.database.JSONDatabase(
                    path_workload=workload_json_file,
                    path_tuning_record=candidates_json_file,
                )
    except Exception:
        print(f'Read database err in file {idx}, {workload_json_file}')
        if os.path.exists(workload_json_file):
            os.remove(workload_json_file)
        if os.path.exists(candidates_json_file):
            os.remove(candidates_json_file)
        return (idx, {}, {})
    
    
    tuning_records = database.get_all_tuning_records()
    keys=['kind', 'inputs', 'attrs', 'decisions', 'outputs']
    for tuning_record in tuning_records:
        Listdict = Sch2Listdict(tuning_record.trace)
        for dict in Listdict:
            if dict['kind'][0] == 'EnterPostproc': continue # remove EnterPostproc Sch
            # if dict['kind'][0] == 'Annotate': continue # remove Annotate Sch
            # if dict['kind'][0] == 'Unannotate': continue # remove Unannotate Sch
            embedding_len = 0
            for key in keys:
                if key == 'kind':
                    if dict[key][0] not in local_kind_embedding:   
                        local_kind_embedding[dict[key][0]] = []
                else:
                    local_str_embedding.update(\
                        {item: -1.0 for item in dict[key] if item not in local_str_embedding\
                            and is_float(item)[0]==False})
                    embedding_len += len(dict[key])
            local_max_embedding_len = max(local_max_embedding_len, embedding_len)
        local_max_seq_len = max(local_max_seq_len, len(Listdict))
        
    return (idx, local_kind_embedding, local_str_embedding, \
        local_max_embedding_len, local_max_seq_len)
 
def make_embedding(measured_records, save_path, save_name):
    str_embedding: Dict[str, float] = {} # embed BlockRV, LoopRV, Expr and String 
    kind_embedding: Dict[str, List] = {} # one-hot kind
    max_embedding_len = -1                  # without onehot
    max_seq_len = -1
    workload_json_files = []
    candidates_json_files = []
    
    model_dirs = sorted(glob.glob(os.path.join(measured_records, "*")))[0:]
    for model_dir in model_dirs:
        wjf = sorted(glob.glob(os.path.join(model_dir, "*_workload.json")))
        workload_json_files.extend(wjf)
        candidates_json_files.extend([file.replace("_workload.json", "_candidates.json") \
            for file in wjf])
    
    multiprocessing_pool = multiprocessing.Pool(processes=32)
    que_res_list = [] 
    local_embeddings = []
    for idx, (workload_json_file, candidates_json_file) in \
        enumerate(zip(workload_json_files, candidates_json_files)):
                que_res_list.append(multiprocessing_pool.apply_async(\
                    handle_file, args=(idx, workload_json_file, candidates_json_file)))
    
    multiprocessing_pool.close()
    multiprocessing_pool.join()
    
    for que_res in que_res_list:
        local_embeddings.append(que_res.get())
    
    # for idx, (workload_json_file, candidates_json_file) in \
    #     enumerate(zip(workload_json_files, candidates_json_files)):
    #             que_res_list.append(handle_file(idx, workload_json_file, candidates_json_file))
    
    # for que_res in que_res_list:
    #     local_embeddings.append(que_res)
    
    for local_embedding in local_embeddings:
        local_kind_embedding: Dict = local_embedding[1]
        local_str_embedding: Dict = local_embedding[2]
        local_max_embedding_len: int = local_embedding[3]
        local_max_seq_len: int = local_embedding[4]
        for k, v in local_kind_embedding.items():
            if k not in kind_embedding:
                kind_embedding[k] = v
        for k, v in local_str_embedding.items():
            if k not in str_embedding:
                str_embedding[k] = v
        max_embedding_len = max(max_embedding_len, local_max_embedding_len)
        max_seq_len = max(max_seq_len, local_max_seq_len)
        
                
    kind_embedding = dict(sorted(kind_embedding.items()))
    str_embedding = dict(sorted(str_embedding.items()))
    
    i = 0
    for k, v in kind_embedding.items():
        one_hot = [0] * len(kind_embedding)
        one_hot[i] = 1
        kind_embedding[k] = one_hot
        i+=1
    i = 0
    shuffle = []
    for k, v in str_embedding.items():
        shuffle.append(k)
    str_embedding = {}
    import random
    random.shuffle(shuffle)
    for k in shuffle:
        str_embedding[k] = i
        i += 1
    max_embedding_len = max_embedding_len + len(kind_embedding)  # with onehot
    max_seq_len = max_seq_len
    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump((kind_embedding, str_embedding, max_embedding_len, max_seq_len), f)
        
def get_embedding_table(save_path: str, save_name: str):
    kind_embedding, str_embedding, max_embedding_len, max_seq_len = \
        pickle.load(open(os.path.join(save_path, save_name), 'rb'))
    return (kind_embedding, str_embedding, max_embedding_len, max_seq_len)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    args = parser.parse_args()
    intrin_name = args.intrin_name
    measured_path = os.path.join(TensorizeSet_dir, f'{intrin_name}_dataset/measured_records')
    table_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/tlp_table')
    table_save_name = f'tlp_{intrin_name}_embedding_table.pkl'
    # make_embedding(measured_path, table_save_path, table_save_name)
    kind_embedding, str_embedding, max_embedding_len, max_seq_len \
        = get_embedding_table(table_save_path, table_save_name)
    print(kind_embedding)
    print(str_embedding)
    print(max_embedding_len)
    print(max_seq_len)