TensorizeSet_dir = '/path/to/your/dataset'
TRASH_PRINT = '/path/to/your/tmp_dir'
dataset_name = 'all_dataset'
HI_name = 'test'
HI_shape = 'm16n16k16nn'
HI_dtype = 'float16'
        
# NETWORK_INFO_FOLDER = f'{TensorizeSet_dir}{HI_name}_{HI_shape}_{HI_dtype}_dataset/network_info'
# TASK_INFO_FOLDER = f'{TensorizeSet_dir}{HI_name}_{HI_shape}_{HI_dtype}_dataset/task_info'
# TO_MEASURE_PROGRAM_FOLDER = f'{TensorizeSet_dir}{HI_name}_{HI_shape}_{HI_dtype}_dataset/to_measure_programs'
# MEASURE_RECORD_FOLDER = f'{TensorizeSet_dir}{HI_name}_{HI_shape}_{HI_dtype}_dataset/measured_records'

NETWORK_INFO_FOLDER = f'{TensorizeSet_dir}/{dataset_name}/network_info'
TASK_INFO_FOLDER = f'{TensorizeSet_dir}/{dataset_name}/task_info'
TO_MEASURE_PROGRAM_FOLDER = f'{TensorizeSet_dir}/{dataset_name}/to_measure_programs'
MEASURE_RECORD_FOLDER = f'{TensorizeSet_dir}/{dataset_name}/measured_records'

TENSORCORE_TARGET = 'nvidia/nvidia-a100'
CASCADELAKE_VNNI_TARGET = "llvm -mcpu=cascadelake -num-cores 4"
SKYLAKE_AVX512_TARGET = "llvm -mcpu=skylake-avx512 -num-cores 4"
ARM_NEON_TARGET = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+dotprod -num-cores 4"
DEFAULT_LLVM = "llvm -num-cores 4"
DEFAULT_CUDA = "cuda"
# TARGET = 'llvm'

# TensorCore float16, int8
# AVX512, VNNI int8
HW_CAST_DTYPE = 'float16'


INIT_POPULATION_SIZE = 512
NUM_SAMPLES_PER_TASK = 1000
# measured
MEASURE_BATCH_SIZE = 500
MAX_RETRY_PER_TASK = 70

NUM_TRIALS_PER_ITER = 64
MAX_TRIALS_PER_TASK = 400
FILE_GROUP = 0

SPACE_KIND = "cuda-tensorcore"



def get_target_from_Target(Target):
    target = None
    if Target == TENSORCORE_TARGET:
        target = "tensorcore"
    elif Target == SKYLAKE_AVX512_TARGET:
        target = "avx512"
    elif Target == CASCADELAKE_VNNI_TARGET:
        target = "vnni"
    elif Target == ARM_NEON_TARGET:
        target = "neon"
    elif Target == DEFAULT_LLVM: 
        target = "llvm"
    elif Target == DEFAULT_CUDA:
        target = "cuda"
    return target

def get_Target_from_model_name(model_name):
    target = None
    if "tensorcore" in model_name:
        target = TENSORCORE_TARGET
    elif "avx512" in model_name:
        target = SKYLAKE_AVX512_TARGET
    elif "vnni" in model_name:
        target = CASCADELAKE_VNNI_TARGET
    elif "neon" in model_name or "sdot" in model_name:
        target = ARM_NEON_TARGET
    elif "llvm" in model_name:
        target = DEFAULT_LLVM
    elif "cuda" in model_name:
        target = DEFAULT_CUDA
    return target

TaskSet_161616nn_fp16 = set()
TaskSet_161616nt_fp16 = set()
TaskSet_83216nn_fp16 = set()
TaskSet_83216nt_fp16 = set()
TaskSet_32816nn_fp16 = set()
TaskSet_32816nt_fp16 = set()
TaskSet_avx512_int8 = set()
TaskSet_vnni_int8 = set()
TaskSet_neon_int8 = set()
TaskSet_sdot_int8 = set()
TaskSet_llvm = set()
TaskSet_cuda = set()

def get_TaskSet_from_model_name(model_name):
    TaskSet = None
    if "161616nn" in model_name and "float16" in model_name:
        TaskSet = TaskSet_161616nn_fp16
    elif "161616nt" in model_name and "float16" in model_name:
        TaskSet = TaskSet_161616nt_fp16
    elif "83216nn" in model_name and "float16" in model_name:
        TaskSet = TaskSet_83216nn_fp16
    elif "83216nt" in model_name and "float16" in model_name:
        TaskSet = TaskSet_83216nt_fp16
    elif "32816nn" in model_name and "float16" in model_name:
        TaskSet = TaskSet_32816nn_fp16
    elif "32816nt" in model_name and "float16" in model_name:
        TaskSet = TaskSet_32816nt_fp16
    elif "avx512" in model_name and "int8" in model_name:
        TaskSet = TaskSet_avx512_int8
    elif "vnni" in model_name and "int8" in model_name:
        TaskSet = TaskSet_vnni_int8
    elif "neon" in model_name and "int8" in model_name:
        TaskSet = TaskSet_neon_int8
    elif "sdot" in model_name and "int8" in model_name:
        TaskSet = TaskSet_sdot_int8
    elif "llvm" in model_name:
        TaskSet = TaskSet_llvm
    elif "cuda" in model_name:
        TaskSet = TaskSet_cuda
    return TaskSet