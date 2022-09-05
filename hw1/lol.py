import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
from collections import Counter
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


def get_device(force_cpu, status=True):
    if not force_cpu and torch.backends.mps.is_available():
      device = torch.device('mps')
      if status:
          print("Using MPS")
    elif not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    counts_per_book = {}
    for line, book in train:
        line = preprocess_string(line)
        padded_len = 2  # start/end
        if book not in counts_per_book:
            counts_per_book[book] = Counter()
        for word in line.lower().split():
            if len(word) > 0:
                word_list.append(word)
                padded_len += 1
                counts_per_book[book][word] += 1
        padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        counts_per_book
    )


def build_output_table(train):
    books = set()
    for _, book in train:
        books.add(book)
    books_to_index = {b: i for i, b in enumerate(books)}
    index_to_books = {books_to_index[b]: b for b in books_to_index}
    return books_to_index, index_to_books


def get_tfidf_weights(counts_per_book, vocab_to_index, books_to_index):
    tfidf_ws = np.zeros((len(books_to_index), len(vocab_to_index)))
    for w in vocab_to_index:
        doc_frequency = sum([counts_per_book[b][w] for b in books_to_index])
        for b in books_to_index:
            tfidf_ws[books_to_index[b], vocab_to_index[w]] = counts_per_book[b][w] / doc_frequency if doc_frequency > 0 else 0
    return tfidf_ws


def tfidf_preds(tfidf_ws, inputs):
    preds = []
    for enc_line in inputs:
        book_weights = [sum([tfidf_ws[bidx][enc_line[idx]] for idx in range(len(enc_line))]) for bidx in range(tfidf_ws.shape[0])]
        preds.append(np.argmax(book_weights))
    return preds


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # Lowercase string
    s = s.lower()
    return s


def process_book_dir(d, max_per_book=None):
    processed_text_lines = []
    n_files = 0
    for root, dirs, fns in os.walk(d):
        for fn in fns:
            if fn.split('.')[-1] == 'txt':
                n_files += 1
                book_title = fn.split('.')[0]
                with open(os.path.join(d, fn), 'r') as f:
                    new_lines = [[preprocess_string(s), book_title] for s in f.readlines() if len(preprocess_string(s)) > 0]
                    if max_per_book is not None and max_per_book < len(new_lines):
                        new_lines = new_lines[:max_per_book]
                    processed_text_lines.extend(new_lines)
    processed_text_lines = [[line, label] for (line, label) in processed_text_lines if len(line.split()) > 0]
    print('read in %d lines from %d files in directory %s' % (len(processed_text_lines), n_files, d))
    return processed_text_lines


def create_train_val_splits(all_lines, prop_train=0.8):
    books = set([b for l, b in all_lines])
    train_lines = []
    val_lines = []
    for b in books:
        lines = [all_lines[idx] for idx in range(len(all_lines)) if all_lines[idx][1] == b]
        val_idxs = np.random.choice(list(range(len(lines))), size=int(len(lines)*prop_train + 0.5), replace=False)
        train_lines.extend([lines[idx] for idx in range(len(lines)) if idx not in val_idxs])
        val_lines.extend([lines[idx] for idx in range(len(lines)) if idx in val_idxs])
    return train_lines, val_lines


def encode_data(data, v2i, seq_len, b2i):
    n_lines = len(data)
    n_books = len(b2i)
    x = np.zeros((n_lines, seq_len), dtype=np.int32)
    y = np.zeros((n_lines), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for txt, book in data:
        txt = preprocess_string(txt)
        x[idx][0] = v2i["<start>"]
        jdx = 1
        for word in txt.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        y[idx] = b2i[book]
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, y


def main(args):
    # Some hyperparameters
    validate_every_n_epochs = 10
    max_epochs = args.epochs
    minibatch_size = 256
    learning_rate = 0.0001
    embedding_dim = args.emb_dim

    # Read in all data
    all_lines = process_book_dir(args.books_dir)

    # Create train/val splits
    train_lines, val_lines = create_train_val_splits(all_lines, prop_train=0.8)

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff, cpb = build_tokenizer_table(train_lines, vocab_size=args.voc_k)
    books_to_index, index_to_books = build_output_table(train_lines)

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_y = encode_data(train_lines, vocab_to_index, len_cutoff, books_to_index)
    # train_y_weight = np.array([1. / (sum([train_np_y[jdx] == idx for jdx in range(len(train_np_y))]) / len(train_np_y)) for idx in range(len(books_to_index))], dtype=np.float32)
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    val_np_x, val_np_y = encode_data(val_lines, vocab_to_index, len_cutoff, books_to_index)
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))

    # Get TFIDF weights from training data.
    tfidf_ws = get_tfidf_weights(cpb, vocab_to_index, books_to_index)

    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=minibatch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=minibatch_size)

    # Run a training loop with an embedded validation loop.
    device = get_device(True)
    model = BookIdet(device, len(vocab_to_index), len_cutoff, len(books_to_index), embedding_dim)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(train_y_weight))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        books_preds = []
        books_labels = []
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            model.train()
            books_out = model(inputs)

            # calculate the loss and train accuracy and perform backprop
            loss = criterion(books_out, labels[:].long())

            # Step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute metrics
            books_preds_ = books_out.argmax(-1)
            books_preds.extend(books_preds_.cpu().numpy())
            # books_preds.extend(tfidf_preds(tfidf_ws, inputs)) # TFIDF
            books_labels.extend(labels[:].cpu().numpy())

        train_accuracy = accuracy_score(books_preds, books_labels)
        print('...Epoch %d train accuracy: %.4f' % (epoch, train_accuracy))

        # Run validation check.
        if epoch % validate_every_n_epochs == 0 or epoch == max_epochs - 1:
            val_books_lines = []
            val_books_preds = []
            val_books_labels = []
            for (inputs, labels) in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                model.eval()
                books_out = model(inputs)

                # compute metrics
                books_preds_ = books_out.argmax(-1)
                val_books_preds.extend(books_preds_.cpu().numpy())
                # val_books_preds.extend(tfidf_preds(tfidf_ws, inputs)) # TFIDF
                val_books_labels.extend(labels[:].cpu().numpy())
                val_books_lines.extend(inputs[:].cpu().numpy())

            val_accuracy = accuracy_score(val_books_preds, val_books_labels)
            print('Epoch %d validation accuracy: %.4f' % (epoch, val_accuracy))

            # show some examples
            k_random_examples = np.random.choice(list(range(len(val_books_lines))), size=10, replace=False)
            for idx in k_random_examples:
                print(' input:\t' + ' '.join([index_to_vocab[val_books_lines[idx][jdx]] for jdx in range(len(val_books_lines[idx]))
                                              if val_books_lines[idx][jdx] > 0]))
                print('   true book:\t%s' % index_to_books[val_books_labels[idx]])
                print('   pred book:\t%s' % index_to_books[val_books_preds[idx]])

            # print the confusion matrix
            cm = np.zeros((len(books_to_index), len(books_to_index)))
            for idx in range(len(val_books_labels)):
                cm[val_books_labels[idx]][val_books_preds[idx]] += 1
            print('; '.join(['%d:%s' % (idx, index_to_books[idx]) for idx in range(len(books_to_index))]))
            print('tr/pr\t  ' + '    |   '.join(['%d' % idx for idx in range(len(books_to_index))]))
            for idx in range(len(books_to_index)):
                print(str(idx) + '\t' + ' | '.join(['%.4f' % (cm[idx][jdx] / len(val_books_lines)) for jdx in range(len(books_to_index))]))
            print('\ndist')
            print('true\t' + ' | '.join(['%.4f' % (sum(cm[idx]) / len(val_books_lines)) for idx in range(len(books_to_index))]))
            print('pred\t' + ' | '.join(['%.4f' % (sum([cm[jdx][idx] for jdx in range(len(cm))]) / len(val_books_lines)) for idx in range(len(books_to_index))]))
            # print('weight\t' + ' | '.join(['%.4f' % train_y_weight[idx] for idx in range(len(books_to_index))]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_dir", type=str, help="books directory", required=True)
    parser.add_argument("--voc_k", type=int, help="vocabulary size", required=True)
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", required=True)
    parser.add_argument("--epochs", type=int, help="max epochs to run", required=False, default=100)
    args = parser.parse_args()
    main(args)