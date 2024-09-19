import argparse
import glob
import json
import os
from typing import List, Set
import tempfile

from tqdm import tqdm  # type: ignore
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.utils import shash2hex
from tvm.ir import load_json
import tvm.tir.tensor_intrin
import common
from common import get_TaskSet_from_model_name

task_cache_dir = common.TASK_INFO_FOLDER
candidate_cache_dir = common.TO_MEASURE_PROGRAM_FOLDER
# target: str = common.TARGET
init_population_size = common.INIT_POPULATION_SIZE
num_samples_per_task = common.NUM_SAMPLES_PER_TASK
num_trials_per_iter = common.NUM_TRIALS_PER_ITER
max_trials_per_task = common.MAX_RETRY_PER_TASK
max_retry_per_task = common.MAX_RETRY_PER_TASK
file_group = common.FILE_GROUP
# space_kind = common.SPACE_KIND
trash_dir: str = common.TRASH_PRINT

tensorcore_target = common.TENSORCORE_TARGET
avx_target = common.SKYLAKE_AVX512_TARGET
vnni_target = common.CASCADELAKE_VNNI_TARGET
neon_target = common.ARM_NEON_TARGET
defaultllvm_target = common.DEFAULT_LLVM
defaultcuda_target = common.DEFAULT_CUDA

def avx512_vnni_neon_test_build(states: List[tvm.tir.Schedule], target):
    print("enter building test ...")
    result = []
    builder = ms.builder.LocalBuilder(max_workers=32, timeout_sec = 3600)
    batch_builder_inputs:List[ms.builder.BuilderInput] = []
    for state in states:
        batch_builder_inputs.append(ms.builder.BuilderInput(state.mod, tvm.target.Target(target)))
    builder_res:List[ms.builder.BuilderResult] = builder.build(batch_builder_inputs)
    err_cnt = 0
    for i, br in enumerate(builder_res):
        if br.error_msg is None:
            result.append(states[i])
        else:
            err_cnt += 1
    print(f"build test, err_cnt:\t{err_cnt}, result_size:\t {len(result)}")
    return result, err_cnt


def sample_candidates(task, task_name, model_name, target):
    """Randomly sample candidates for a task and save the candidates in the given directory.

    Parameters
    ----------
    task : IRModule
        The initial ir module used for generating the search space.
    task_name : str
        The name of the task.
    model_name : str
        The name of the model.

    Returns
    -------
    None
    """
    print(f'dump programs for {model_name}, {task_name} ...')
        
    candidate_path = os.path.join(
        candidate_cache_dir, model_name, task_name + "_candidates.json"
    )
    workload_path = os.path.join(
        candidate_cache_dir, model_name, task_name + "_workload.json"
    )
    
    TaskSet:Set = get_TaskSet_from_model_name(model_name)
    TaskModHash = shash2hex(task)
    if TaskModHash in TaskSet:
        if os.path.exists(workload_path):
            os.remove(workload_path)
            os.remove(candidate_path)
        if os.path.exists(str(workload_path).replace(".json", ".skip")):
            os.remove(str(workload_path).replace(".json", ".skip"))
            os.remove(str(candidate_path).replace(".json", ".skip"))
        if os.path.exists(str(workload_path).replace(".json", ".fail")):
            os.remove(str(workload_path).replace(".json", ".fail"))
            os.remove(str(candidate_path).replace(".json", ".fail"))
        print(f"duplicate {model_name}, {task_name} ...")
        return None
    else:
        TaskSet.add(TaskModHash)

    if os.path.exists(workload_path)\
        or os.path.exists(str(workload_path).replace(".json", ".skip"))\
        or os.path.exists(str(workload_path).replace(".json", ".fail")):
            print(f'{model_name}, {task_name} already exits, skipping...' )
            return None
    if 'winograd' in task_name:
        print(f'winograd skipping ...')
        return None
    if 'tanh' in task_name:
        print(f'fast_tanh skipping ...')
        return None
    sample_init_population = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation"
    )
    evolve_with_cost_model = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )
    
    if target == tensorcore_target:
        # sch_rules = "cuda-tensorcore"
        if "161616nn" in model_name:
            sch_rules = "cuda-tensorcore-161616nn"
            success_flag = "wmma_sync_16x16x16"
        elif "161616nt" in model_name:
            sch_rules = "cuda-tensorcore-161616nt"
            success_flag = "wmma_sync_16x16x16"
        elif "83216nn" in model_name:
            sch_rules = "cuda-tensorcore-83216nn"
            success_flag = "wmma_sync_8x32x16"
        elif "83216nt" in model_name:
            sch_rules = "cuda-tensorcore-83216nt"
            success_flag = "wmma_sync_8x32x16"
        elif "32816nn" in model_name:
            sch_rules = "cuda-tensorcore-32816nn"
            success_flag = "wmma_sync_32x8x16"
        elif "32816nt" in model_name:
            sch_rules = "cuda-tensorcore-32816nt"
            success_flag = "wmma_sync_32x8x16"
        postprocs = "cuda-tensorcore"
        mutator = "cuda-tensorcore"
    elif target == avx_target:
        sch_rules = "avx512"
        postprocs = "avx512"
        mutator = "avx512"
        success_flag = "avx512"
    elif target == vnni_target:
        sch_rules = "vnni"
        postprocs = "vnni"
        mutator = "vnni"
        success_flag = "vnni"
    elif target == neon_target:
        sch_rules = "neon"
        postprocs = "neon"
        mutator = "neon"
        success_flag = "neon"
    elif target == defaultllvm_target:
        sch_rules = "llvm"
        postprocs = "llvm"
        mutator = "llvm"
        success_flag = "block"
        
    generator = PostOrderApply(sch_rules=sch_rules, postprocs=postprocs, mutator_probs=mutator)
    strategy = ms.search_strategy.EvolutionarySearch(init_measured_ratio=0.0,
                                                     genetic_num_iters=3,)
    
    context = ms.TuneContext(
        mod=task,
        target=tvm.target.Target(target),
        space_generator=generator,
        search_strategy=strategy,
        task_name=task_name,
        num_threads="physical",
    )
    # context.initialize()
    try:
        design_space = context.generate_design_space()
        
        is_success = False
        for it in design_space: 
            if success_flag in str(it.trace): 
                is_success = True
            
        if is_success:
            database = ms.database.JSONDatabase(
                path_workload=workload_path,
                path_tuning_record=candidate_path,
            )
        else:
            # with tempfile.NamedTemporaryFile('w') as temp_file:
            database = ms.database.JSONDatabase(
                path_workload=str(workload_path).replace(".json", ".skip"),
                path_tuning_record=str(candidate_path).replace(".json", ".skip")
            )
            database.commit_workload(context.mod)
            print(f'{model_name}, {task_name}, tensorize skip, skipping...')
            return None
        
        context.pre_tuning(
            max_trials=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=design_space,
            database=database,
            cost_model=ms.cost_model.RandomModel(),  # type: ignore
        )

        all_states: List[tvm.tir.Schedule] = []
        all_states_set: set[str] = set()
        num_retry, itr = 0, 0
        
        # states = sample_init_population(strategy, init_population_size)
        while len(all_states) < num_samples_per_task and num_retry < max_retry_per_task:
            states = sample_init_population(strategy, init_population_size)
            states: List[tvm.tir.Schedule] = \
                evolve_with_cost_model(strategy, states, int(len(states) * 19))
            
            if target == avx_target or target == vnni_target or target == neon_target \
               or "161616nt" in model_name or "83216nt" in model_name or "32816nt" in model_name :
                states, err_cnt = avx512_vnni_neon_test_build(states, target)
            else:
                states = states
                err_cnt = 0
            
            for state in states:
                if str(state.trace) not in all_states_set:
                    all_states_set.add(str(state.trace))
                    all_states.append(state)
            if len(states) == 0 or err_cnt != 0:
                # states = sample_init_population(strategy, init_population_size)
                num_retry += 1
            else:
                num_retry = 0
            print(f"iter: {itr}, number of states sampled: {len(all_states)}")
            itr += 1
        all_states = all_states[: num_samples_per_task]

        workload = ms.database.Workload(context.mod)
        database.commit_workload(context.mod)
        for state in all_states:
            database.commit_tuning_record(ms.database.TuningRecord(state.trace, workload))
    except Exception:
        if os.path.exists(workload_path):
            os.remove(workload_path)
        if os.path.exists(candidate_path):
            os.remove(candidate_path)
        # with tempfile.NamedTemporaryFile('w') as temp_file:
        database = ms.database.JSONDatabase(
            path_workload=str(workload_path).replace(".json", ".fail"),
            path_tuning_record=str(candidate_path).replace(".json", ".fail")
        )
        database.commit_workload(context.mod)
        print(f'{model_name}, {task_name}, tensorize fail, skipping...')
        return None
        



def main():
    if not os.path.isdir(task_cache_dir):
        raise Exception("Please provide a correct task cache dir.")
    try:
        os.makedirs(candidate_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {candidate_cache_dir} cannot be created successfully.")

    # task_paths = sorted(glob.glob(os.path.join(task_cache_dir, "*.json")))[
    #     file_group * 10 : (file_group + 1) * 10
    # ]
    task_paths = sorted(glob.glob(os.path.join(task_cache_dir, "*.json")))[
        0 :
    ]
    # task_paths = sorted(glob.glob(os.path.join(task_cache_dir, "*.json")))
    
    print(f"Selected models: {task_paths}")
    for num, task_path in enumerate(task_paths):
        print(f"Processing model {num} ...")
        model_name = task_path.split("/")[-1][len("tasks-") :][: -len("_extracted_tasks.json")]
        
        target = common.get_Target_from_model_name(model_name)
        if target == None: continue
        
        model_names = []                      # TO Support tensorcore intrinsics, others ignore
        if target == tensorcore_target:
            # parts = model_name.split("-")
            # parts[3] += ",161616nn"
            # model_names.append("-".join(parts))
            
            parts = model_name.split("-")
            parts[3] += ",161616nt"
            model_names.append("-".join(parts))
            
            # parts = model_name.split("-")
            # parts[3] += ",83216nn"
            # model_names.append("-".join(parts))
            
            parts = model_name.split("-")
            parts[3] += ",83216nt"
            model_names.append("-".join(parts))
            
            # parts = model_name.split("-")
            # parts[3] += ",32816nn"
            # model_names.append("-".join(parts))
            
            parts = model_name.split("-")
            parts[3] += ",32816nt"
            model_names.append("-".join(parts))
            
        else:
            model_names.append(model_name)
        
        for model_name in model_names:
            os.makedirs(os.path.join(candidate_cache_dir, model_name), exist_ok=True)
            with open(task_path, "rb") as file:
                tasks = file.readlines()
            for task_str in tqdm(tasks):
                task_name, task_mod, _, _ = json.loads(task_str)
                task_mod = load_json(json.dumps(task_mod))
                sample_candidates(task_mod, task_name, model_name, target)


if __name__ == "__main__":
    main()