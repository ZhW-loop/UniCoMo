# DEFAULT to CUDA TensorCore. Only reference, NO USE.
import sys 
sys.path.append("TensorizeSet_app/taskops")
import argparse
import glob
import json
import os
from tqdm import tqdm  # type: ignore

import tvm
from tvm import meta_schedule as ms
from tvm.ir import save_json
from taskops import relay_workload
from tvm.runtime import load_param_dict
import common

model_cache_dir = common.NETWORK_INFO_FOLDER
task_cache_dir = common.TASK_INFO_FOLDER
# target: str = common.TARGET
trash_dir: str = common.TRASH_PRINT

tensorcore_target = common.TENSORCORE_TARGET
avx_target = common.SKYLAKE_AVX512_TARGET
vnni_target = common.CASCADELAKE_VNNI_TARGET
neon_target = common.ARM_NEON_TARGET
defaultllvm_target = common.DEFAULT_LLVM
defaultcuda_target = common.DEFAULT_CUDA

def extract_and_save_tasks(cache_file:str, target):
    """Extract tuning tasks and cache the nonspatial ones in the given directory.

    Parameters
    ----------
    cache_file : str
        The filename of the cached model.

    Returns
    -------
    None
    """
    task_cache_file = cache_file.split(".")[0].split("-")
    task_cache_file[0] = "tasks"
    # task_cache_file.insert(4, tvm.target.Target(target).kind.name)
    task_cache_file.insert(4, common.get_target_from_Target(target))
    task_cache_file = "-".join(task_cache_file)
    to_trash_filename = task_cache_file
    task_cache_file += "_extracted_tasks.json"
    
    task_cache_path = os.path.join(
        task_cache_dir, task_cache_file
    )
    
    if os.path.exists(task_cache_path):
        print(f'{task_cache_file} already exits, skipping...')
        return None
    
    mod, params_bytearray, _ = relay_workload._load_cache(model_cache_dir, cache_file)
    params = load_param_dict(params_bytearray)
    try:
        extracted_tasks = ms.relay_integration.extract_tasks(
            mod, 
            target=target, 
            params=params,
            opt_level=3
            )
    except tvm.error.TVMError as error:
        print(str(error))
        return
    
    is_spatial = tvm.get_global_func("tir.schedule.IsSpatialPrimFunc")
    with open(task_cache_path, "w", encoding="utf8") as file, \
        open(f"{trash_dir}{to_trash_filename}.trash", "w") as f:
        for i, task in enumerate(extracted_tasks):
            subgraph = task.dispatched[0]
            high_mod = task.mod
            prim_func = subgraph[subgraph.get_global_vars()[0]]
            if not is_spatial(prim_func):
                subgraph_str = save_json(subgraph)
                high_mod_str = save_json(high_mod)
                json_obj = [task.task_name, json.loads(subgraph_str), json.loads(high_mod_str), task.weight]
                json_str = json.dumps(json_obj)
                assert "\n" not in json_str, "Failed to generate single line string."
                if i == len(extracted_tasks) - 1:
                    file.write(json_str)
                else:
                    file.write(json_str + "\n")
                f.write(str(f'{task.task_name}\n'))
                f.write(str(f'{high_mod}\n'))
                f.write(str(f'{task.weight}\n'))
                f.write(str(subgraph))
                f.write("\n\n") 



def main():
    target_list = [tensorcore_target, avx_target, vnni_target, neon_target]
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=0)
    args = parser.parse_args()
    target = target_list[0]
    
    if not os.path.isdir(model_cache_dir):
        raise Exception("Please provide a correct model cache dir.")
    try:
        os.makedirs(task_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {task_cache_dir} cannot be created successfully.")

    paths = glob.glob(os.path.join(model_cache_dir, "*.json"))  # pylint: disable=invalid-name
    for path in tqdm(paths):
        filename = path.split("/")[-1]
        # target = tensorcore_target
        # target = avx_target
        # target = neon_target
        extract_and_save_tasks(filename, target)


if __name__ == "__main__":
    main()