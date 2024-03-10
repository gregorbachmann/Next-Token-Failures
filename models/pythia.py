import torch
import torch.nn as nn
from models.config import PythiaConfig
from models.lib import Attention, MLP, LayerNorm
from models.base_model import Transformer
from utils.load import load_pythia


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, layer_idx, rotary=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=None):
        residual = x

        x_att = self.attn(self.ln_1(x), cache)
        x_mlp = self.mlp(self.ln_2(x))
        x = x_att + x_mlp + residual

        return x


class Pythia(Transformer):
    def __init__(self, config):
        super().__init__(config, block=Block)

    @classmethod
    def from_pretrained(cls, model_type, teacherless_token=None):
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'pythia-70m-deduped': dict(n_layers=6, n_heads=8, n_embd=512),  # 124M params
            'pythia-160m-deduped': dict(n_layers=12, n_heads=12, n_embd=768),  # 350M params
            'pythia-410m-deduped': dict(n_layers=24, n_heads=16, n_embd=1024),  # 774M params
            'pythia-1b-deduped': dict(n_layers=16, n_heads=8, n_embd=2048),  # 1558M params
            'pythia-1.4b-deduped': dict(n_layers=24, n_heads=16, n_embd=2048)
        }[model_type]
        print("forcing vocab_size=50304, block_size=2048, bias=True")
        config_args[
            'vocab_size'] = 50304
        config_args['block_size'] = 2048
        config_args['bias'] = True  # always True for Pythia model checkpoints
        config_args['teacherless_token'] = teacherless_token

        # create a from-scratch initialized Pythia model with right dimensions
        config = PythiaConfig(**config_args)
        model = Pythia(config)
        sd = model.state_dict()

        # init a huggingface/transformers model to get the weights
        from transformers import GPTNeoXForCausalLM
        model_hf = GPTNeoXForCausalLM.from_pretrained("EleutherAI/" + model_type, revision="step43000")
        sd_hf = model_hf.state_dict()

        # Match the weights
        sd = load_pythia(sd, sd_hf, config)
        model.load_state_dict(sd, strict=True)

        return model


if __name__ == "__main__":
    import types
    from tokenizing import get_tokenizer

    args = types.SimpleNamespace()
    args.model = 'pythia-70m-deduped'
    tokenizer = get_tokenizer(args)

    model = Pythia.from_pretrained(model_type=args.model)
    model.eval()
    text = "Hello, I am"
    idx = torch.tensor(tokenizer.encode(text), dtype=torch.int64).unsqueeze(0)
    #model.set_cache(device='cpu')
    out = model.generate(idx, max_new_tokens=54, top_k=1)
    print(tokenizer.decode(out.numpy().squeeze()))
