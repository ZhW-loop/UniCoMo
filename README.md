## UniCoMo: A Unified Learning-Based Cost Model for Tensorized Program Tuning
### Environment Setup

```shell
git clone --recursive https://github.com/ZhW-loop/UniCoMo.git
cd UniCoMo
conda env create --file conda/build-environment.yaml
conda activate tvm-build
mkdir build && cd build
cp ../cmake/config.cmake . # set USE_LLVM ON, set USE_CUDA ON
cmake ..
make -j32
pip3 install numpy decorator attrs
pip3 install tornado
pip3 install tornado psutil 'xgboost>=1.1.0' cloudpickle
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### TensorizeSet

Configuration common.py.

```python
TensorizeSet_dir = '/path/to/your/dataset'
TRASH_PRINT = '/path/to/your/tmp_dir'
```

Dump neural network and subgraph.

```shell
# For tensor core with float16
python dump_relay_models.py && python dump_relay2task.py
# For avx512 or neon with int8
python dump_op2task.py
```

Dump tensorized programs in search space. Tensorized programs on the same semantics share the same schedules (but can also be different), while those on different semantics have different schedules (and cannot be the same).

```shell
python dump_programs.py && python programs_avx2vnni.py && python programs_neon2sdot.py
```

Dump tensorized programs with latency label.

```sh
python dump_measured.py 
```

Sort the dataset files.

```sh
cd '/path/to/your/dataset'
mv ./measured_records/*tensorcore161616nn* tensorcore161616nn_dataset/measured_records
mv ./measured_records/*tensorcore161616nt* tensorcore161616nt_dataset/measured_records
mv ./measured_records/*tensorcore32816nn* tensorcore32816nn_dataset/measured_records
mv ./measured_records/*tensorcore32816nt* tensorcore32816nt_dataset/measured_records
mv ./measured_records/*tensorcore83216nn* tensorcore83216nn_dataset/measured_records
mv ./measured_records/*tensorcore83216nt* tensorcore83216nt_dataset/measured_records
mv ./measured_records/*avx512* avx512_dataset/measured_records
mv ./measured_records/*vnni* vnni_dataset/measured_records
mv ./measured_records/*neon* neon_dataset/measured_records
mv ./measured_records/*sdot* sdot_dataset/measured_records
```



### AST Feature Mining with Attention

Configuration common.py.

```python
TensorizeSet_dir = '/path/to/your/dataset'
TRASH_PRINT = '/path/to/your/tmp_dir'
```

Make TLP Embedding Table.

```sh
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
python embedding_table.py --intrin_name tensorcore161616nn
```

Make TLP DataSet. Encoding scheduling primitives as NLP tasks.

```shell
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
python tlp_make_dataset.py --intrin_name tensorcore161616nn
```

Transfer TLP to tensorized programs and train it on TensorizeSet, called TDTLP.

```shell
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
python tlp_train.py --intrin_name tensorcore161616nn
```

Test TDTLP with top-k score.

```shell
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
# --model_id 0-n_epoch
python tlp_eval.py --intrin_name tensorcore161616nn --model_id 37
```

Draw Schedule Attention Matrix.

```shell
# build cal_attention
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cal_attention.cpp -o cal_attention`python3-config --extension-suffix`

# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
# --model_id 0-n_epoch
python get_attention_matrix.py --intrin_name tensorcore161616nn --model_id 37
python draw_ScheduleAttention.py --intrin_name tensorcore161616nn
```



### Unified Feature Representation

In the implementation, unified feature representation is called tensorize notation (TN). Feature extraction and encoding based on the unified feature representation are implemented in 

```tex
src/meta_schedule/feature_extractor/per_block_feature.cc
```

Configuration common.py.

```python
TensorizeSet_dir = '/path/to/your/dataset'
TRASH_PRINT = '/path/to/your/tmp_dir'
```

Make TN DataSet with unified feature representation.

```sh
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
python TN_make_dataset.py --intrin_name tensorcore161616nn
```

Train UniCoMo.

```shell
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
python TN_train.py --intrin_name tensorcore161616nn
```

Test UniCoMo with top-k score.

```shell
# --intrin_name tensorcorexxx, avx512, vnni, neon, sdot
# --model_id 0-n_epoch
python TN_eval.py --intrin_name tensorcore161616nn --model_id 37
```



### Life Long Learning

Generate a task dataset through catDataLoaderLL.py to simulate the cross-semantic scenario. The following configuration can generate the dataset of Task 7 in the paper. Following a similar configuration, generate tasks 9, 0, 1, 2, and 4 in the paper.

```python
new_task_id = 7
LL_task_save_name = f"task{new_task_id}_train.pkl.DataLoader"
pkl_dataloader_paths = ['tensorcore161616nn_train_and_val.pkl.DataLoader',
                        'avx512_train_and_val.pkl.DataLoader',
                        'sdot_train_and_val.pkl.DataLoader']
```

Configuration common.py.

```python
TensorizeSet_dir = '/path/to/your/dataset'
TRASH_PRINT = '/path/to/your/tmp_dir'
```

Train UniCoMo with life long learning.

```sh
python LL_train.py --last_task_id -1 --last_task_model_id -1 --new_task_id 9 --train_type standard
python LL_train.py --last_task_id 9 --last_task_model_id 37 --new_task_id 0 --train_type LL
python LL_train.py --last_task_id 0 --last_task_model_id 37 --new_task_id 1 --train_type LL
python LL_train.py --last_task_id 1 --last_task_model_id 37 --new_task_id 2 --train_type LL
python LL_train.py --last_task_id 2 --last_task_model_id 37 --new_task_id 4 --train_type LL
```

