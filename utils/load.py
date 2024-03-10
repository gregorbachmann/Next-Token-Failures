import torch


def remap(key, mapping):
    return mapping[key] if key in mapping else key


def load_gpt(state_dict, hf_state_dict):
    mapping = {
        'wte': 'embed_tokens',
        'wpe': 'pos_encoding',
        'h': 'layers',
        'ln_f': 'final_layernorm',
        'self_attn': 'attn',
        'c_proj': 'proj',
        'c_fc': 'expand',
    }

    check_keys_loaded = {key: False for key in state_dict}
    for key, val in hf_state_dict.items():
        mapped_key = key
        if key.startswith('transformer.'):
            mapped_key = key.split('transformer.')[1]
        mapped_key = '.'.join([remap(s, mapping) for s in mapped_key.split('.')])

        if mapped_key in state_dict or mapped_key.endswith('c_attn.weight') or mapped_key.endswith('c_attn.bias'):
            if mapped_key.endswith('c_attn.weight') or mapped_key.endswith('c_attn.bias'):
                dim = hf_state_dict[key].shape[-1] // 3
                with torch.no_grad():
                    state_dict['queries_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., :dim].t())
                    check_keys_loaded['queries_linear'.join(mapped_key.split('c_attn'))] = True
                    state_dict['keys_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., dim:2*dim].t())
                    check_keys_loaded['keys_linear'.join(mapped_key.split('c_attn'))] = True
                    state_dict['values_linear'.join(mapped_key.split('c_attn'))]\
                        .copy_(hf_state_dict[key][..., 2*dim:3*dim].t())
                    check_keys_loaded['values_linear'.join(mapped_key.split('c_attn'))] = True
            else:
                try:
                    if mapped_key.endswith('mlp.expand.weight') or mapped_key.endswith('proj.weight'):
                        with torch.no_grad():
                            state_dict[mapped_key].copy_(hf_state_dict[key].t())
                    else:
                        with torch.no_grad():
                            state_dict[mapped_key].copy_(hf_state_dict[key])
                    check_keys_loaded[mapped_key] = True
                except RuntimeError:
                    print(key, 'does not match in shape')
        else:
            print(key, 'was not found')

    for key, val in check_keys_loaded.items():
        if not val:
            print(key, 'was not loaded')

    return state_dict


def load_pythia(state_dict, hf_state_dict, config):
    mapping = {
        'embed_in': 'embed_tokens',
        'embed_out': 'lm_head',
        'input_layernorm': 'ln_1',
        'post_attention_layernorm': 'ln_2',
        'final_layer_norm': 'final_layernorm',
        'attention': 'attn',
        'dense': 'proj',
        'dense_h_to_4h': 'expand',
        'dense_4h_to_h': 'proj',
    }

    check_keys_loaded = {key: False for key in state_dict}
    check_keys_hf_loaded = {key: False for key in hf_state_dict}
    for key, val in hf_state_dict.items():
        mapped_key = key
        if key.startswith('gpt_neox.'):
            mapped_key = key.split('gpt_neox.')[1]
        mapped_key = '.'.join([remap(s, mapping) for s in mapped_key.split('.')])

        if mapped_key in state_dict or mapped_key.endswith('query_key_value.weight') or mapped_key.endswith(
                'query_key_value.bias'):
            if mapped_key.endswith('query_key_value.weight') or mapped_key.endswith('query_key_value.bias'):
                with torch.no_grad():
                    # Have to apply annoying reshapes to qkv weights because Pythia does weird reshapes...
                    head_dim = config.n_embd // config.n_heads
                    hf_state_dict[key] = hf_state_dict[key].reshape((config.n_heads, 3 * head_dim, -1))
                    q = hf_state_dict[key][:, :head_dim, ...].reshape((config.n_embd, -1)).squeeze()
                    k = hf_state_dict[key][:, head_dim:2 * head_dim, ...].reshape((config.n_embd, -1)).squeeze()
                    v = hf_state_dict[key][:, 2 * head_dim:, ...].reshape((config.n_embd, -1)).squeeze()
                    state_dict['queries_linear'.join(mapped_key.split('query_key_value'))].copy_(q)
                    check_keys_loaded['queries_linear'.join(mapped_key.split('query_key_value'))] = True
                    check_keys_hf_loaded[key] = True
                    state_dict['keys_linear'.join(mapped_key.split('query_key_value'))].copy_(k)
                    check_keys_loaded['keys_linear'.join(mapped_key.split('query_key_value'))] = True
                    check_keys_hf_loaded[key] = True
                    state_dict['values_linear'.join(mapped_key.split('query_key_value'))].copy_(v)
                    check_keys_loaded['values_linear'.join(mapped_key.split('query_key_value'))] = True
                    check_keys_hf_loaded[key] = True
            else:
                try:
                    with torch.no_grad():
                        state_dict[mapped_key].copy_(hf_state_dict[key])
                    check_keys_loaded[mapped_key] = True
                    check_keys_hf_loaded[key] = True
                except RuntimeError:
                    print(key, 'does not match in shape')
        else:
            print(key, 'was not found')

    for key, val in check_keys_loaded.items():
        if not val:
            print(key, 'was not loaded')
    for key, val in check_keys_hf_loaded.items():
        if not val:
            print(key, 'was not loaded')

    return state_dict
