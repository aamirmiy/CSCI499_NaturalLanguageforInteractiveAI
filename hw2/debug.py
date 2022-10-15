from data_utils import *
from torch.utils.data import TensorDataset, DataLoader
# processed_text_lines = process_book_dir('/home/aamirmiy/CSCI499_NaturalLanguageforInteractiveAI/hw2/books', max_per_book=None)

# (
#     vocab_to_index,
#     index_to_vocab,
#     suggested_padding_len,
# ) = build_tokenizer_table(processed_text_lines, vocab_size=3000)

# encoded_sentences, lens = encode_data(
#     processed_text_lines,
#     vocab_to_index,
#     suggested_padding_len
# )

# print(len(encoded_sentences))

# train, val = train_val_split(encoded_sentences)
# # print(val[0])
# train_set = collate(train)
# val_set = collate(val)

# print(train_set[0])
# print(val_set[0])
sentences = process_book_dir('/home/aamirmiy/CSCI499_NaturalLanguageforInteractiveAI/hw2/books', max_per_book=None)

# build one hot maps for input and output
(
    vocab_to_index,
    index_to_vocab,
    suggested_padding_len,
) = build_tokenizer_table(sentences, vocab_size=3000)

# create encoded input and output numpy matrices for the entire dataset and then put them into tensors
encoded_sentences, lens = encode_data(
    sentences,
    vocab_to_index,
    suggested_padding_len,
)

train, val = train_val_split(encoded_sentences)
train_x, train_y= collate(train)
val_x, val_y = collate(val)


train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

# ================== TODO: CODE HERE ================== #
# Task: Given the tokenized and encoded text, you need to
# create inputs to the LM model you want to train.
# E.g., could be target word in -> context out or
# context in -> target word out.
# You can build up that input/output table across all
# encoded sentences in the dataset!
# Then, split the data into train set and validation set
# (you can use utils functions) and create respective
# dataloaders.
# ===================================================== #

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32,drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32,drop_last=True)

print(train_loader.shape())
print(train_loader[0])