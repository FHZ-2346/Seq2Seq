import torch
from Data.Vocab import Vocab


def tokenize(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def build_array(lines, vocab: Vocab):
    eos = vocab["<eos>"]
    array = [vocab[l] + [eos] for l in lines]
    return array


def unify_array(array, vocab: Vocab, num_steps):

    def truncate_pad(line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps]  # 截断
        return line + [padding_token] * (num_steps - len(line))  # 填充

    array_unified = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in array])
    valid_len = (array_unified != vocab['<pad>']).type(torch.int32).sum(1)
    return array_unified, valid_len


def load_data_iter(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
    return data_iter
