from tlp_train import SegmentDataLoader, AttentionModule
import torch, sys, argparse
import pickle, numpy as np, os, random
from typing import List
from tlp_embedding import Embedding
from d2l_common import show_heatmaps
import cal_attention # type: ignore
from common import TensorizeSet_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def one_hot2index(one_hot: List):
    return one_hot.index(1)

def get_kind_embedding(table_save_path, table_save_name):
    embedding = Embedding(table_save_path, table_save_name).kind_embedding
    print(embedding)
    return embedding

def get_InputwithAttention(model_file: str, 
                           datasets_file: str, 
                           is_test: bool,
                           device: str = 'cuda:0'):
    
    inputs = []
    attensions = []
    with open(model_file, 'rb') as f:
        model: AttentionModule = pickle.load(f).to(device)
    model.eval()
    print(model.step_size)
    print(model.fea_size)
    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)
    
    datas = []
    if is_test:
        for file_vec in datasets:
            workload_key, line_vecs = file_vec
            datas.extend(line_vecs)
        data_loader = SegmentDataLoader(datas, 4000, False)
        del datasets
    else:
        # datas = random.sample(datasets, 7777)
        # del datasets
        data_loader:SegmentDataLoader = datasets[0]
        data_loader.datas_steps = data_loader.datas_steps[37777: 377777]
        data_loader.labels = data_loader.labels[37777: 377777]
        data_loader.min_latency = data_loader.min_latency[37777: 377777]
        data_loader.number = len(data_loader.datas_steps)
        del datasets
    
    for batch_datas_steps, batch_labels in data_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        preds = model(batch_datas_steps)
        inputs.append(batch_datas_steps.detach().cpu().numpy())
        attensions.append(model.attention_matrix.detach().cpu().numpy())
    
    inputs = np.concatenate(inputs, axis=0)
    attensions = np.concatenate(attensions, axis=0)
    return inputs, attensions

def load_ScheduleAttention(save_path):
    with open(save_path, 'rb') as f:
        results = pickle.load(f)
    return results
def save_ScheduleAttention(ScheduleAttention, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(ScheduleAttention, f)


def get_ScheduleAttention(table_save_path, table_save_name, model_file, datasets_file,
                          is_test,
                          save_path):
    # if os.path.exists(save_path):
    #     print(f"load ScheduleAttention from cache {save_path}...")
    #     return load_ScheduleAttention(save_path)
    
    inputs, attentions = get_InputwithAttention(model_file, datasets_file, is_test)
    print(inputs.shape, attentions.shape)
    assert inputs.shape[0] == attentions.shape[0]
    assert inputs.shape[1] == attentions.shape[1]
    assert inputs.shape[1] == attentions.shape[2]
    kind_emb = get_kind_embedding(table_save_path, table_save_name)
    results = np.zeros((len(kind_emb), len(kind_emb)))
    results_cnt = np.zeros((len(kind_emb), len(kind_emb)))
    # inputs2KindIndex = np.zeros((inputs.shape[0], inputs.shape[1], 1))
    # inputs2KindIndex[:, :, 0] = np.argmax(inputs[:, :, 0:len(kind_emb)], axis=2)
    # inputs2KindIndex = inputs2KindIndex.astype(np.int32)
    
    tmp_sum = np.sum(inputs[:, :, 0:len(kind_emb)], axis=-1)
    inputs2KindIndex = np.where(tmp_sum == 0, -1, np.argmax(inputs[:, :, 0:len(kind_emb)], axis=-1))
    inputs2KindIndex = np.expand_dims(inputs2KindIndex, axis=-1)
    inputs2KindIndex = inputs2KindIndex.astype(np.int32)
    print(inputs2KindIndex.shape)
    
    
    # for batch in range(attentions.shape[0]):
    #     for i in range(attentions.shape[1]):
    #         for j in range(attentions.shape[2]):
    #             ii = inputs2KindIndex[batch][i][0]
    #             jj = inputs2KindIndex[batch][j][0]
    #             results[ii][jj] += attentions[batch][i][j]
    #             results_cnt[ii][jj] += 1
    print("Enter cpp calAttention ...")
    cal_attention.calAttention(attentions, inputs2KindIndex, results, results_cnt)
    print("Finish cpp calAttention ...")
    
    # results_cnt[results_cnt < 1] = 1e9
    # results = results / (results_cnt + 1)
    
    results = results / inputs.shape[0]
    
    # import torch.nn.functional as F
    # results = torch.Tensor(results).reshape(-1)
    # results = F.softmax(results, dim=0)
    # results = results.reshape((len(kind_emb), len(kind_emb)))
    
    save_ScheduleAttention(results, save_path)
    return results


def draw_ScheduleAttention(ScheduleAttention):
    ScheduleAttention = torch.Tensor(ScheduleAttention)
    assert ScheduleAttention.shape[0] == ScheduleAttention.shape[1]
    ScheduleAttention = \
        ScheduleAttention.reshape((1, 1, ScheduleAttention.shape[0], ScheduleAttention.shape[1]))
    show_heatmaps(ScheduleAttention, "Keys", "Queries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    parser.add_argument("--model_id", type=str, default='37')
    args = parser.parse_args()
    intrin_name = args.intrin_name
    model_id = args.model_id
    
    measured_path = os.path.join(TensorizeSet_dir, f'{intrin_name}_dataset/measured_records') 
    
    dataset_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/tlp_dataset') 
    train_save_name = f'tlp_{intrin_name}_train_and_val.pkl.DataLoader'
    eval_save_name = f'tlp_{intrin_name}_test.pkl'
    
    model_save_path = os.path.join(TensorizeSet_dir, f'tlp_records/tlp_models/{intrin_name}') 
    
    task_info_path = os.path.join(TensorizeSet_dir, f'task_info') 
    # datasets_file = os.path.join(measured_records, 'tlp_339_test.pkl')
    # USE test_dataset
    datasets_file = os.path.join(dataset_save_path, eval_save_name)
    
    # USE train_dataset
    # datasets_file = os.path.join(dataset_save_path, train_save_name)
    
    model_file = os.path.join(model_save_path, f'tlp_model_{model_id}.pkl')
    
    table_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/tlp_table')
    table_save_name = f'tlp_{intrin_name}_embedding_table.pkl'
    
    attention_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/attention_matrix') 
    attention_save_name = f'ScheduleAttention_{intrin_name}.pkl'
    
    ScheduleAttention_save_path = os.path.join(attention_save_path, attention_save_name)
    ScheduleAttention = get_ScheduleAttention(table_save_path, table_save_name, model_file, datasets_file, 
                                              True,
                                                # False,
                                              ScheduleAttention_save_path)
    # print(ScheduleAttention)