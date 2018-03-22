import os
import zipfile
import torchtext
from torchtext import data, datasets
import sys


"""
Data preprocessing: save dictionaries and dataset
Returns batches: x (batch_size, window_size) |
word indices; y (batch_size, 1) | word indices of next word
Return R matrix (embedding matrix)
Return w2i, i2w dictionary (vocab)
"""


def load_ptb(ptb_path='data.zip', ptb_dir='data', bptt_len=5, batch_size=1,
             gpu=False, reuse=False, repeat=False, shuffle=True):
    print("Loading Data")
    if (not reuse) or (not os.path.exists(ptb_dir)):
        f = zipfile.ZipFile(ptb_path, 'r')
        f.extractall('.')
        f.close()

    DEV = 0 if gpu else -1

    text_field = data.Field(lower=True, batch_first=True,
                            init_token="<s>", eos_token="</s>")
    train = datasets.LanguageModelingDataset(os.path.join(
            ptb_dir, 'train.txt'), text_field, newline_eos=False)
    val = datasets.LanguageModelingDataset(os.path.join(
            ptb_dir, 'valid.txt'), text_field, newline_eos=False)
    test = datasets.LanguageModelingDataset(os.path.join(
            ptb_dir, 'test.txt'), text_field, newline_eos=False)

    train_iter, val_iter, test_iter = data.BPTTIterator.splits(
            (train, val, test),
            batch_size=batch_size,
            bptt_len=bptt_len,
            device=DEV,
            repeat=repeat,
            shuffle=shuffle)

    text_field.build_vocab(
            train, min_freq=250, vectors=torchtext.vocab.GloVe(name='6B', dim=100))

    return train_iter, val_iter, test_iter, text_field
