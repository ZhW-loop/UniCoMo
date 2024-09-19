# DEFAULT to CUDA TensorCore. Only reference, NO USE.

import sys 
sys.path.append("TensorizeSet_app/taskops")
import argparse
import os
from typing import List, Tuple
import threading

from tqdm import tqdm  # type: ignore
from taskops import relay_workload
import common
import multiprocessing
model_cache_dir = common.NETWORK_INFO_FOLDER

# pylint: disable=too-many-branches
def _build_dataset() -> List[Tuple[str, List[int]]]:
    network_keys = []
    for name in [
        "resnet_18",
        "resnet_50",
        # "resnet_101",
        "mobilenet_v2",
        # "wide_resnet_50",
        # "resnext_50",
        # "densenet_121",
    ]:
        for batch_size in [1, 8, 16]:
            for image_size in [224, 240, 256]:
                for layout in ["NHWC"]:
                    network_keys.append((name, [batch_size, 3, image_size, image_size], layout))
    
    # for name in ["resnext_50"]:
    #     for batch_size in [1, 8, 16, 32]:
    #         for image_size in [224, 240, 256]:
    #             for layout in ["NCHW"]:
    #                 network_keys.append((name, [batch_size, 3, image_size, image_size], layout))
                 
    # # inception-v3
    for name in ["inception_v3"]:
        for batch_size in [1, 8, 16]:
            for image_size in [299]:
                for layout in ["NHWC"]:
                    network_keys.append((name, [batch_size, 3, image_size, image_size], layout))
    # resnet3d
    for name in ["resnet3d_18"]:
        for batch_size in [1, 8, 16]:
            for image_size in [112, 128, 144]:
                for layout in ["NHWC"]:
                    network_keys.append((name, [batch_size, 3, image_size, image_size, 16], layout))
    # bert
    for name in ["bert_tiny", 
                 "bert_base", 
                 "bert_medium", 
                 "bert_large",
                 ]:
        for batch_size in [1, 8, 16]:
            for seq_length in [64, 128, 256]:
                for layout in ["None"]:
                    network_keys.append((name, [batch_size, seq_length], layout))
    # dcgan
    # for name in ["dcgan"]:
    #     for batch_size in [1, 4, 8]:
    #         for image_size in [64]:
    #             network_keys.append((name, [batch_size, 3, image_size, image_size]))
    return network_keys

def poolmap(x): relay_workload.get_network(name = x[0], input_shape = x[1], cache_dir = x[2], layout = x[3])
def main():
    try:
        os.makedirs(model_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {model_cache_dir} cannot be created successfully.")
    keys = _build_dataset()
    for name, input_shape, layout in tqdm(keys):
        relay_workload.get_network(name=name, 
                                    input_shape=input_shape, 
                                    cache_dir=model_cache_dir,
                                    layout=layout)
    # pool = multiprocessing.Pool(processes=8)
    # map_list = [(name, input_shape, model_cache_dir) for name, input_shape in keys]
    # pool.map(poolmap, map_list)
    # pool.close()
    # pool.join()
    
    # threads = []
    # for name, input_shape in tqdm(keys):
    #     thread = threading.Thread(target=poolmap, args=([(name, input_shape, model_cache_dir)]))
    #     thread.start()
    #     threads.append(thread)
    # for thread in threads:
    #     thread.join()

if __name__ == "__main__":
    main()
