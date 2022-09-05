#import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


class BookIdet(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        input_len,
        n_books,
        embedding_dim
    ):
        super(BookIdet, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_books = n_books

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        # linear layer
        self.fc = torch.nn.Linear(embedding_dim, n_books)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        maxpooled_embeds = self.maxpool(embeds)
        out = self.fc(maxpooled_embeds).squeeze(1)  # squeeze out the singleton length dimension that we maxpool'd over
        
        return out

from utils import (
    get_device,
    read_episodes,
    flatten_list,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)

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
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data, vocab_size = 1000)
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

    return train_loader, val_loader

trainer,validloader = setup_dataloader('lang_to_sem_data.json')

# Display image and label.
train_features, train_labels = next(iter(validloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
label = train_labels[:10,0]
label1 = train_labels[:10,1]
print(label)
print(label1)
