# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring

import argparse
import glob
import os

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.builder import BuilderInput
from tvm.meta_schedule.builder import BuilderResult
from tvm.meta_schedule.runner import LocalRunner
from tvm.meta_schedule.runner import RunnerInput
from tvm.meta_schedule.runner import RunnerFuture
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.runner import EvaluatorConfig
import tvm.tir.tensor_intrin
import common
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TVM_NUM_THREADS"] = "4"
candidate_cache_dir = common.TO_MEASURE_PROGRAM_FOLDER
result_cache_dir = common.MEASURE_RECORD_FOLDER
# target = common.TARGET
batch_size = common.MEASURE_BATCH_SIZE
tensorcore_target = common.TENSORCORE_TARGET
avx_target = common.SKYLAKE_AVX512_TARGET
vnni_target = common.CASCADELAKE_VNNI_TARGET

# reset_gpu = tvm.get_global_func("device_api.cuda_reset")
# pylint: disable=too-many-locals
def measure_candidates(workload_path,
                       candidate_path,
                       builder: ms.builder.LocalBuilder, 
                       runner: ms.runner.LocalRunner,
                       model_name,
                       workload_name,
                       record_name,
                       target):
    """Send the candidates to builder and runner for distributed measurement,
    and save the results in a new json database.

    Parameters
    ----------
    database : JSONDatabase
        The database for candidates to be measured.
    builder : Builder
        The builder for building the candidates.
    runner : Runner
        The runner for measuring the candidates.

    Returns
    -------
    None
    """
    task_name = workload_name[:-len("_workload.json")]
    print(f"mearsure {model_name}, {task_name} ...")
    if os.path.exists(os.path.join(result_cache_dir, model_name, workload_name)) \
        or os.path.exists(os.path.join(result_cache_dir, model_name, record_name)):
            print(f"{model_name}, {task_name} already exists, skipping ...")
            return None
            
    
    database = ms.database.JSONDatabase(
                path_workload=workload_path,
                path_tuning_record=candidate_path,
            )
    
    candidates, runner_results, build_fail_indices, run_fail_indices = [], [], [], []
    tuning_records = database.get_all_tuning_records()
    
    for record in tqdm(tuning_records):
        candidates.append(record.as_measure_candidate())
        
    for idx in tqdm(range(0, len(candidates), batch_size)):
        batch_candidates: list[MeasureCandidate] \
            = candidates[idx : idx + batch_size]
        
        
        batch_builder_inputs = []
        for candidate in batch_candidates:
            batch_builder_inputs.append(BuilderInput(candidate.sch.mod, 
                                                Target(target)))
        batch_builder_results: list[BuilderResult] \
            = builder.build(batch_builder_inputs)
        
        # for builder_result in batch_builder_results:
        #     assert builder_result.error_msg is None
        
        print("Build finishing ...")
        
        batch_runner_inputs = []
        n_build_errors = 0
        for i in range(len(batch_builder_results)):
            if batch_builder_results[i].error_msg is not None:
                n_build_errors += 1
                continue
            batch_runner_inputs.append(RunnerInput(batch_builder_results[i].artifact_path,
                                                Target(target).kind.name,
                                                batch_candidates[i].args_info))
        
        
        
        temp_futures: list[RunnerFuture] = runner.run(batch_runner_inputs)
        batch_runner_futures = []
        if n_build_errors == 0:
            batch_runner_futures.extend(temp_futures)
        else:
            j = 0
            for i in range(len(batch_builder_results)):
                builder_result = batch_builder_results[i]
                if builder_result.error_msg is not None:
                    batch_runner_futures.append(RunnerFuture(
                        f_done=lambda: True,
                        f_result=lambda msg=builder_result.error_msg: RunnerResult(None, msg)
                    ))
                else:
                    batch_runner_futures.append(temp_futures[j])
                    j += 1
        
        
        batch_runner_results: list[RunnerResult] = []
        for runner_future in batch_runner_futures:
            batch_runner_results.append(runner_future.result())
        
        # for runner_result in batch_runner_results:
        #     assert runner_result.error_msg is None
        
        runner_results.extend(batch_runner_results) 
        for i, result in enumerate(batch_builder_results):
            if result.error_msg is None:
                ms.utils.remove_build_dir(result.artifact_path)
            else:
                build_fail_indices.append(i + idx)

    print(f"\ncommit {model_name}, {task_name} measured record ...")
    
    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(result_cache_dir, model_name, workload_name),
        path_tuning_record=os.path.join(result_cache_dir, model_name, record_name),
    )
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
    
    for i, (record, result) in enumerate(zip(tuning_records, runner_results)):
        if result.error_msg is None:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    run_secs=[v.value for v in result.run_secs],
                    target=Target(target),
                )
            )
        else:
            run_fail_indices.append(i)
    fail_indices_name = workload_name.replace("_workload.json", "_failed_indices.txt")
    if len(build_fail_indices) != 0 or len(run_fail_indices) != 0:
        with open(
            os.path.join(result_cache_dir, model_name, fail_indices_name), "w", encoding="utf8"
        ) as file:
            file.write(f"build fail indices: {len(build_fail_indices)}\n")
            file.write(" ".join([str(n) for n in build_fail_indices]))
            file.write("\n")
            file.write(f"run fail indices: {len(run_fail_indices)}\n")
            file.write(" ".join([str(n) for n in run_fail_indices]))
            file.write("\n")

    print(f"Failed number of builds: {len(build_fail_indices)}\n")
    print(f"Failed number of runs: {len(run_fail_indices)}\n")


def main():
    builder = LocalBuilder(timeout_sec = 3600.0)
    evaluator_config = EvaluatorConfig(
        number=3,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )
    runner = LocalRunner(evaluator_config=evaluator_config)
    if not os.path.isdir(candidate_cache_dir):
        raise Exception("Please provide a correct candidate cache dir.")
    try:
        os.makedirs(result_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {result_cache_dir} cannot be created successfully.")
    model_dirs = sorted(glob.glob(os.path.join(candidate_cache_dir, "*")))
    
    model_dirs = model_dirs[0: ]
    
    for model_dir in model_dirs:
        model_name = model_dir.split("/")[-1]
        target = common.get_Target_from_model_name(model_name)
        if target == None: continue
        # if target == avx_target or target == vnni_target:
        #     continue
    
        os.makedirs(os.path.join(result_cache_dir, model_name), exist_ok=True)
        all_tasks = sorted(glob.glob(os.path.join(model_dir, "*.json")))
        workload_paths = []
        for path in all_tasks:
            if path.endswith("_workload.json"):
                workload_paths.append(path)
        for workload_path in tqdm(workload_paths):
            candidate_path = workload_path.replace("_workload.json", "_candidates.json")
            measure_candidates(workload_path, candidate_path, 
                               builder, runner, 
                               model_name, 
                               workload_path.split("/")[-1],
                               candidate_path.split("/")[-1],
                               target)


if __name__ == "__main__":
    main()
