import os
import re
import json
import gensim
import tqdm
import torch
import numpy as np
from collections import Counter
from spacy.lang.en import English


def process_book_dir(d, max_per_book=None):
    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.max_length = 3293518  # we aren't doing any heavy parsing or anything; set based on biggest book
    processed_text_lines = []
    n_files = 0
    for root, dirs, fns in os.walk(d):
        for fn in fns:
            if fn.split(".")[-1] == "txt":
                n_files += 1
                book_title = fn.split(".")[0]
                with open(os.path.join(d, fn), "r") as f:
                    new_lines = [s for s in f.readlines()]
                    if max_per_book is not None and max_per_book < len(new_lines):
                        new_lines = new_lines[:max_per_book]
                    entire_book = " ".join(new_lines)
                    doc = nlp(entire_book)
                    sentences = list(doc.sents)
                    sentences = [preprocess_string(str(s)) for s in sentences]
                    processed_text_lines.extend(
                        [[s, book_title] for s in sentences if len(s) > 0]
                    )
    processed_text_lines = [
        [line, label] for (line, label) in processed_text_lines if len(line.split()) > 0
    ]
    print(
        "read in %d lines from %d files in directory %s"
        % (len(processed_text_lines), n_files, d)
    )
    return processed_text_lines


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    # Lowercase string
    s = s.lower()
    return s


def build_tokenizer_table(train, vocab_size):
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
        int(np.max(padded_lens)),  # we don't need a cutoff for vanilla LM
    )


def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.smart_open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )


def encode_data(data, v2i, seq_len):
    num_insts = len(data)
    x = np.zeros((num_insts, seq_len), dtype=np.int32)
    lens = np.zeros((num_insts, 1), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for sent, source in data:
        x[idx][0] = v2i["<start>"]
        jdx = 1
        for word in sent.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    if len(sent.split()) >= seq_len:
                        n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        lens[idx][0] = jdx
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d sentences at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d sentences without regard to order" % idx)

    return x, lens


def collate_cbow(batch,CBOW_N_WORDS,MAX_SEQUENCE_LENGTH):
    batch_input, batch_output = [], [] #CBOW_N_WORDS is the context window
    for text in batch:
        text_tokens_ids = text

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence[CBOW_N_WORDS]
            token_id_sequence = np.delete(token_id_sequence,CBOW_N_WORDS)
            batch_input.append(token_id_sequence)
            batch_output.append(output)

    batch_input = torch.tensor(np.array(batch_input), dtype=torch.long)
    batch_output = torch.tensor(np.array(batch_output), dtype=torch.long)
    return batch_input, batch_output


def train_val_split(encoded_sentences): #train_val split 80:20
    train_set = encoded_sentences[:int(len(encoded_sentences)*0.8)]
    val_set = encoded_sentences[int(len(encoded_sentences)*0.8):]
    return train_set, val_set