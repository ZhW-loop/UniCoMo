import torch, pickle, os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from tlp_embedding import Embedding
from common import TensorizeSet_dir
import argparse
def draw_Attention(ScheduleAttention_path, table_save_path, table_save_name):
    with open(ScheduleAttention_path, 'rb') as f:
        ScheduleAttention = pickle.load(f)
    embedding = Embedding(table_save_path, table_save_name).kind_embedding
    variables = []
    labels = []
    for k, v in embedding.items():
        variables.append(k)
        labels.append(k)

    df = pd.DataFrame(ScheduleAttention, columns=variables, index=labels)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fontdict_x = {'rotation': 87, 'fontsize': 7, } 
    fontdict_y = {'fontsize': 7}
    ax.set_xticklabels([''] + list(df.columns), fontdict = fontdict_x)
    ax.set_yticklabels([''] + list(df.index), fontdict = fontdict_y)

    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    args = parser.parse_args()
    intrin_name = args.intrin_name
    
    attention_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/attention_matrix') 
    attention_save_name = f'ScheduleAttention_{intrin_name}.pkl'
    ScheduleAttention_save_path = os.path.join(attention_save_path, attention_save_name)

    table_save_path = os.path.join(TensorizeSet_dir, 'tlp_records/tlp_table') 
    table_save_name = f'tlp_{intrin_name}_embedding_table.pkl'
    draw_Attention(ScheduleAttention_save_path, table_save_path, table_save_name)