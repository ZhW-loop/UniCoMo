import sys 
sys.path.append("TensorizeSet_app/taskops") 
from typing import Union, Set, List
from taskops import net_ops
import json
import os
from tqdm import tqdm  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.ir import save_json
import common
model_cache_dir = common.NETWORK_INFO_FOLDER
task_cache_dir = common.TASK_INFO_FOLDER
trash_dir: str = common.TRASH_PRINT

tensorcore_target = common.TENSORCORE_TARGET
avx_target = common.SKYLAKE_AVX512_TARGET
vnni_target = common.CASCADELAKE_VNNI_TARGET
neon_target = common.ARM_NEON_TARGET
defaultllvm_target = common.DEFAULT_LLVM

def extract_and_save_tasks(model: Union[net_ops.net, str], target):
    task_cache_file = "tasks"
    task_cache_file = "-".join([task_cache_file, model.model_name])
    to_trash_filename = task_cache_file
    task_cache_file += "_extracted_tasks.json"
    
    task_cache_path = os.path.join(
        task_cache_dir, task_cache_file
    )
    
    if os.path.exists(task_cache_path):
        print(f'{task_cache_file} already exits, skipping...')
        return None
    mod_params_list = model()
    task_name_set: Set[str] = set()
    def get_legal_task_name(task_name):
        i = 0
        task_name = f"{task_name}_{i}"
        while task_name in task_name_set:
            i += 1
            task_name = task_name.replace(f"_{i-1}", f"_{i}")
        task_name_set.add(task_name)
        return task_name
        
    is_spatial = tvm.get_global_func("tir.schedule.IsSpatialPrimFunc")
    with open(task_cache_path, "w", encoding="utf8") as file, \
        open(f"{trash_dir}{to_trash_filename}.trash", "w") as f:
        for i, mp in enumerate(mod_params_list):
            mod = mp[0]
            params = mp[1]
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
            # if(len(extracted_tasks) != 1):
            #     print(model.model_name)
            #     print(mp[0])
            #     task = extracted_tasks[0]
            #     print(task.mod)
            #     task = extracted_tasks[1]
            #     print(task.mod)
            # assert len(extracted_tasks) == 1
            # task = extracted_tasks[0]
            for task in extracted_tasks:
                subgraph = task.dispatched[0]
                high_mod = task.mod
                prim_func = subgraph[subgraph.get_global_vars()[0]]
                if not is_spatial(prim_func):
                    subgraph_str = save_json(subgraph)
                    high_mod_str = save_json(high_mod)
                    task_name = get_legal_task_name(task.task_name)
                    json_obj = [task_name, json.loads(subgraph_str), json.loads(high_mod_str), mp[2]]
                    json_str = json.dumps(json_obj)
                    assert "\n" not in json_str, "Failed to generate single line string."
                    if i == len(mod_params_list) - 1:
                        file.write(json_str)
                    else:
                        file.write(json_str + "\n")
                    f.write(str(f'{task_name}\n'))
                    f.write(str(f'{high_mod}\n'))
                    f.write(str(subgraph))
                    f.write("\n\n") 



def main():
    try:
        os.makedirs(task_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {task_cache_dir} cannot be created successfully.")
        
    all_model: List[net_ops.net] = []
    # model = net_ops.net('resnet_18', (1, 3, 224, 224), None, None, tensorcore_target)
    # all_model.append(model)
    
    # model = net_ops.net('custom', (0, 0, 0, 0), None, None, avx_target)
    # all_model.append(model)
    
    # model = net_ops.net('custom', (0, 0, 0, 0), None, None, vnni_target)
    # all_model.append(model)
    
    # model = net_ops.net('custom', (0, 0, 0, 0), None, None, tensorcore_target)
    # all_model.append(model)
    
    # model = net_ops.net('defaultLLVM', (0, 0, 0, 0), None, None, defaultllvm_target)
    # all_model.append(model)
    
    # densenet
    model = net_ops.net('densenet_121', (1, 3, 224, 224), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 224, 224), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 224, 224), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (1, 3, 240, 240), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 240, 240), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 240, 240), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (1, 3, 256, 256), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 256, 256), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 256, 256), 'NCHW', 'int8', common.SKYLAKE_AVX512_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (1, 3, 224, 224), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 224, 224), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 224, 224), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (1, 3, 240, 240), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 240, 240), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 240, 240), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (1, 3, 256, 256), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (8, 3, 256, 256), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)
    
    model = net_ops.net('densenet_121', (16, 3, 256, 256), 'NCHW', 'int8', common.ARM_NEON_TARGET)
    all_model.append(model)

    # bert
    bert_name = ['bert_base', 'bert_large', 'bert_medium', 'bert_tiny']
    bert_batch = [1, 8, 16]
    bert_seq = [64, 128, 256]
    bert_target = [common.SKYLAKE_AVX512_TARGET, common.ARM_NEON_TARGET]
    
    for name in bert_name:
        for batch in bert_batch:
            for seq in bert_seq:
                for target in bert_target:
                    model = net_ops.net(name, (batch, seq), "None", 'int8', target)
                    all_model.append(model)
    
    # resnet18, 50
    resnet_name = ['resnet_18', 'resnet_50']
    resnet_batch = [1, 8, 16]
    resnet_input = [(3, 224, 224), (3, 240, 240), (3, 256, 256)]
    resnet_target = [common.SKYLAKE_AVX512_TARGET, common.ARM_NEON_TARGET]
    
    for name in resnet_name:
        for batch in resnet_batch:
            for input in resnet_input:
                for target in resnet_target:
                    model = net_ops.net(name, (batch, input[0], input[1], input[2]), "NCHW", 'int8', target)
                    all_model.append(model)
    
    
    # inception_v3
    inception_name = ['inception_v3']
    inception_batch = [1, 8, 16]
    inception_input = [(3, 299, 299)]
    inception_target = [common.SKYLAKE_AVX512_TARGET, common.ARM_NEON_TARGET]
    
    for name in inception_name:
        for batch in inception_batch:
            for input in inception_input:
                for target in inception_target:
                    model = net_ops.net(name, (batch, input[0], input[1], input[2]), "NCHW", 'int8', target)
                    all_model.append(model)
    
    # mobilenet_v2
    mobilenet_name = ['mobilenet_v2']
    mobilenet_batch = [1, 8, 16]
    mobilenet_input = [(3, 224, 224), (3, 240, 240), (3, 256, 256)]
    mobilenet_target = [common.SKYLAKE_AVX512_TARGET, common.ARM_NEON_TARGET]
    for name in mobilenet_name:
        for batch in mobilenet_batch:
            for input in mobilenet_input:
                for target in mobilenet_target:
                    model = net_ops.net(name, (batch, input[0], input[1], input[2]), "NCHW", 'int8', target)
                    all_model.append(model)
    
    # resnet3d_18
    resnet3d_name = ['resnet3d_18']
    resnet3d_batch = [1, 8, 16]
    resnet3d_input = [(16, 3, 112, 112), (16, 3, 128, 128), (16, 3, 144, 144)]
    resnet3d_target = [common.SKYLAKE_AVX512_TARGET, common.ARM_NEON_TARGET]
    for name in resnet3d_name:
        for batch in resnet3d_batch:
            for input in resnet3d_input:
                for target in resnet3d_target:
                    model = net_ops.net(name, (batch, input[0], input[1], input[2], input[3]), "NCHW", 'int8', target)
                    all_model.append(model)
    
    
    for model in all_model:
        extract_and_save_tasks(model, model.target)
    
if __name__ == "__main__":
    main()
    
    
