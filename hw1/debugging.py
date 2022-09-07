#import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


from utils import (
    get_device,
    preprocess_string,
    read_episodes,
    flatten_list,
    build_tokenizer_table,
    build_output_tables,
    #encode_data
)


def encode_data(flat_data, v2i, seq_len, t2i, a2i):
    n_lines = len(flat_data)
    n_target_class = len(t2i)
    n_action_class = len(a2i)
    x = np.zeros((n_lines, seq_len), dtype = np.int32)
    y = np.zeros((n_lines,2), dtype = np.int32)
    #y1 = np.zeros((n_lines), dtype=np.int32)
    #y2 = np.zeros((n_lines), dtype=np.int32)
    
    idx=0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for lines in flat_data:
        instruction = lines[0]
        classes = lines[1]
        instruction = preprocess_string(instruction)
        x[idx][0] = v2i["<start>"]
        jdx=1
        for word in instruction.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len -1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        y[idx][0] = a2i[classes[0]]
        y[idx][1] = t2i[classes[1]]
        #y1[idx] = a2i[classes[0]]
        #y2[idx] = t2i[classes[1]]
        idx += 1
    # print("INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
    # % (n_unks, n_tks, n_unks / n_tks, len(v2i)))
    # print("INFO: cut off %d instances at len %d before true ending"
    # % (n_early_cutoff, seq_len))
    # print("INFO: encoded %d instances without regard to order" % idx)
    return x, y
# train_data, val_data =  read_episodes('lang_to_sem_data.json')
# #print(train_data[0])
# train_data, val_data =  read_episodes('lang_to_sem_data.json')
# vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data, vocab_size = 10000)
# actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)
# train_data = flatten_list(train_data)
# train_np_x, train_np_y =  encode_data(train_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)

# print(train_np_x[0])
# print(train_np_y[0])

# result = flatten_list(train_data)
# print(result[0])

# for input in result:
#     instruction = input[0]
#     labels = input[1]
#     print(instruction)
#     print(labels[0])
#     print(labels[1])
#     break
def setup_dataloader(file):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    train_data, val_data =  read_episodes(file)
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data, vocab_size = 10000)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)
    train_data = flatten_list(train_data)
    val_data = flatten_list(val_data)
    #train_np_x, train_np_y1, train_np_y2 =  encode_data(train_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    train_np_x, train_np_y =  encode_data(train_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    #train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y1), torch.from_numpy(train_np_y2))
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    #val_np_x, val_np_y1, val_np_y2 = encode_data(val_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, len_cutoff, targets_to_index, actions_to_index)
    #val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y1), torch.from_numpy(val_np_y2))
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)

    return train_np_x, train_np_y, train_loader, val_loader, index_to_vocab,index_to_actions,index_to_targets

x,y,trainer,validloader,dic,dic1,dic2 = setup_dataloader('lang_to_sem_data.json')

# Display image and label.
# train_features, train_labels = next(iter(trainer))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_x.shape)
#print(train_features)
for (inputs,labels) in trainer:
    # input  = inputs[1]
    # action = labels[1,0]
    # target = labels[1,1]
    print(inputs.shape)
    print(labels.shape)
    break
# print(input)
# print(action)
# print(target)
# input_list = input.tolist()
# action_list = action.tolist()
# target_list = target.tolist()

# for i in input_list:
#     word = dic[i]
#     print(word)
# print(dic1[action_list])
# print(dic2[target_list])