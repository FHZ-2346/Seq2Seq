import collections


class Vocab:

    def __init__(self, tokens=[], min_freq=0, tokens_reserved=[]):
        if len(tokens) == 0:  # will manuallly load_map_i2t later
            return
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]  # Flatten 2D tokens
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        tokens_valid = [token for token, freq in token_freqs if freq >= min_freq]
        tokens_all = tokens_reserved + ["<unk>"] + tokens_valid
        self.idx2token = list(sorted(set(tokens_all), key=lambda t: t[0]))

        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}

    def load_map_i2t(self, idx2token):
        self.idx2token = idx2token
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}

    @property  # decorator : disguise method as property
    def unk(self):  # Index for the unknown token
        return self.token2idx["<unk>"]
    
    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(token) for token in tokens]
        elif isinstance(tokens, str):
            return self.token2idx.get(tokens, self.unk)
        else:
            raise TypeError("Input Invalid")

    def to_tokens(self, indices):
        if hasattr(indices, "__len__"):
            return [self.idx2token[int(index)] for index in indices]
        elif isinstance(indices, int):
            return self.idx2token[indices]
        else:
            raise TypeError("Input Invalid")
