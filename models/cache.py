import torch


class Cache:
    def __init__(self, config, use_caching=False):
        self.config = config
        self.use_caching = use_caching
        self.dtype = config.dtype

        self.seq_len = config.block_size
        self.max_bsz = config.max_bsz
        self.n_atts = config.n_layers
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.cur_seq_len = {layer_idx: 0 for layer_idx in range(config.n_layers)}
        self.cur_bsz = 0
        # Do not consume memory yet, only after build() is called
        self.key_cache = None
        self.value_cache = None

    def build(self, device):
        self.key_cache = torch.zeros((self.max_bsz, self.seq_len, self.n_atts, self.n_heads, self.head_dim),
                                     device=device, dtype=self.dtype)
        self.value_cache = torch.zeros((self.max_bsz, self.seq_len, self.n_atts, self.n_heads, self.head_dim),
                                       device=device, dtype=self.dtype)

    def update(self, keys, values, layer_idx):
        bsz, cur_seq_len, _, _ = keys.shape
        self.cur_seq_len[layer_idx] = cur_seq_len
        self.cur_bsz = bsz
        self.key_cache[:bsz, :self.cur_seq_len[layer_idx], layer_idx, ...] = keys
        self.value_cache[:bsz, :self.cur_seq_len[layer_idx], layer_idx, ...] = values

    def get(self, layer_idx):
        return (
            self.key_cache[:self.cur_bsz, :self.cur_seq_len[layer_idx], layer_idx, :, :],
            self.value_cache[:self.cur_bsz, :self.cur_seq_len[layer_idx], layer_idx, :, :]
                )

    def empty(self):
        # Set cache back to zero
        self.cur_seq_len = {layer_idx: 0 for layer_idx in range(self.n_atts)}
        self.key_cache.zero_()
        self.value_cache.zero_()

    def delete(self):
        # Free memory completely
        self.cur_seq_len = {layer_idx: 0 for layer_idx in range(self.n_atts)}
        self.key_cache = None
        self.value_cache = None
