import sys 
sys.path.append("TensorizeSet_app") 
import argparse
import glob
import os
import json

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
import tvm.tir.tensor_intrin
import common
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
candidate_cache_dir = common.TO_MEASURE_PROGRAM_FOLDER
# result_cache_dir = common.MEASURE_RECORD_FOLDER
# target = common.TARGET
# batch_size = common.MEASURE_BATCH_SIZE
# tensorcore_target = common.TENSORCORE_TARGET
# avx_target = common.SKYLAKE_AVX512_TARGET
# vnni_target = common.CASCADELAKE_VNNI_TARGET

def candidates_avx2vnni(workload_path, candidate_path, model_name,
                        workload_name, record_name):
    task_name = workload_name[:-len("_workload.json")]
    print(f"change {model_name}, {task_name} ...")
    if os.path.exists(os.path.join(candidate_cache_dir, model_name, workload_name)) \
        or os.path.exists(os.path.join(candidate_cache_dir, model_name, record_name)):
            print(f"{model_name}, {task_name} already exists, skipping ...")
            return None
    os.makedirs(os.path.join(candidate_cache_dir, model_name), exist_ok=True)
    database = ms.database.JSONDatabase(
                path_workload=workload_path,
                path_tuning_record=candidate_path,
            )
    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(candidate_cache_dir, model_name, workload_name),
        path_tuning_record=os.path.join(candidate_cache_dir, model_name, record_name),
    )
    tuning_records = database.get_all_tuning_records()
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
    
    
    def replace_substring_recursive(input_list, substring_to_replace, replacement_substring):
        if isinstance(input_list, list):
            return [replace_substring_recursive(item, substring_to_replace, replacement_substring) for item in input_list]
        elif isinstance(input_list, str):
            return input_list.replace(substring_to_replace, replacement_substring)
        else:
            return input_list
    
    
    for tuning_record in tuning_records:
        tuning_record = tuning_record.as_json()
        assert "dot_16x4_avx512" in str(tuning_record)
        tuning_record = replace_substring_recursive(list(tuning_record), 
                                                    "dot_16x4_avx512", "dot_16x4_vnni")
        assert "dot_16x4_avx512" not in str(tuning_record)
        assert "dot_16x4_vnni" in str(tuning_record)
        tuning_record = ms.database.TuningRecord.from_json(tuning_record, workload)
        new_database.commit_tuning_record(tuning_record)
    

def main():
    if not os.path.isdir(candidate_cache_dir):
        raise Exception("Please provide a correct candidate cache dir.")
    
    model_dirs = sorted(glob.glob(os.path.join(candidate_cache_dir, "*")))
    avx512_model_dirs = []
    
    for md in model_dirs:
        if 'avx512' in str(md):
            avx512_model_dirs.append(md)
    
    for model_dir in avx512_model_dirs:
        model_name: str = model_dir.split("/")[-1]
        model_name = model_name.replace("avx512", "vnni")
        all_tasks = sorted(glob.glob(os.path.join(model_dir, "*.json")))
        workload_paths = []
        for path in all_tasks:
            if path.endswith("_workload.json"):
                workload_paths.append(path)
        for workload_path in tqdm(workload_paths):
            candidate_path = workload_path.replace("_workload.json", "_candidates.json")
            candidates_avx2vnni(workload_path, candidate_path, 
                               model_name, 
                               workload_path.split("/")[-1],
                               candidate_path.split("/")[-1]
                               )


if __name__ == "__main__":
    main()
