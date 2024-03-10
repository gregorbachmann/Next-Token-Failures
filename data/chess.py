import pandas as pd
import json
import torch
from torch.utils.data import Dataset


letters = ['r', 'R', 'k', 'K', 'n', 'N', 'p', 'P', 'b', 'B', 'q', 'Q', ',', '/', ' ', 'w']
letter_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def unroll_state(state):
    unrolled = ''
    for s in state:
        if s in letters:
            unrolled += s
        else:
            unrolled += int(s) * '1,'
            unrolled = unrolled[:-1]

    return unrolled


def unroll(row):
    unrolled = []
    for s in row:
        if s in letters:
            unrolled.append(s)
        else:
            unrolled += int(s) * [1]

    return unrolled


def separate(row):
    out = ''
    for r in row:
        out += r + ','

    return out[:-1]


def collapse(row):
    collapsed = ''
    buffer = 0
    for s in row:
        if s in letters:
            if buffer > 0:
                collapsed += str(buffer)
                buffer = 0
            collapsed += s
        if s not in letters: buffer += 1

    if buffer > 0:
        collapsed += str(buffer)

    return collapsed


def update_board(state, move):
    rows = state.split('/')
    rows[-1], rest = rows[-1].split(' ')[0], ' '.join(rows[-1].split(' ')[1:])
    start_row = 8 - int(move[1])
    start_col = letter_to_col[move[0]]
    end_row = 8 - int(move[3])
    end_col = letter_to_col[move[2]]

    # Remove piece to be moved from corresponding row
    unroll_row = unroll(rows[start_row])
    piece = unroll_row[start_col]
    unroll_row[start_col] = 1
    rows[start_row] = collapse(unroll_row)

    # Add piece to row where it will move
    unroll_row = unroll(rows[end_row])
    unroll_row[end_col] = piece
    rows[end_row] = collapse(unroll_row)

    # Change active color (ignore the other stuff for now)
    if rest[0] == 'b':
        rest = 'w'
    else:
        rest = 'b'

    # Form the new state
    new_state = ''
    for row in rows:
        new_state += separate(row) + '/'

    # Ignore last entry since it's / and merge with rest
    new_state = new_state[:-1] + ' ' + rest

    return new_state


class Chess(Dataset):
    def __init__(self, tokenizer, n_samples=None, data_path=None, device='cpu', eval=False, teacherless_token=None,
                 unrolled=False):
        self.tokenizer = tokenizer
        self.n_samples = n_samples if n_samples is not None else 350000
        self.device = device
        self.eval_mode = eval
        self.teacherless_token = teacherless_token

        with open(data_path) as f:
            data = json.load(f)

        if unrolled:
            data = [(unroll_state(data['prefix'][i]), ''.join(data['target'][i].split(' '))) for i in range(n_samples)
                    if data['target'][i][-1] in numbers]
        else:
            data = [(data['prefix'][i], data['target'][i]) for i in range(n_samples)
                    if data['target'][i][-1] in numbers]
        self.max_length = max([len(data[i][0]) for i in range(len(data))]) + 1

        self.tokenized, self.num_prefix_tokens, self.num_target_tokens = tokenizer.tokenize(data)
        self.num_tokens = self.num_prefix_tokens + self.num_target_tokens

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs
        x = self.tokenized[idx][:-1].clone()
        if self.teacherless_token is not None:
            x[self.num_prefix_tokens:] = torch.tensor(self.teacherless_token)
            x = x.to(self.device)
        # Create targets in the form [-1, ..., -1, 4, 7, 9, 2, ...] where we replace the prefix tokens by -1 so that
        # we can skip their gradient calculation in the loss (double-check if that's correct)
        y = torch.cat([-torch.ones((self.num_prefix_tokens - 1, )),
                       self.tokenized[idx][self.num_prefix_tokens:].clone()])

        return x.to(self.device), y.long().to(self.device)

    def eval(self):
        # Switch to "eval" mode when generating sequences without teacher-forcing
        self.eval_mode = True

    def train(self):
        # Switch back to "train" mode for teacher-forcing
        self.eval_mode = False


if __name__ == '__main__':
    import json
    device = 'cpu'
    mate_in = 2
    """num_moves = (2 * mate_in - 1) * 5
    csv = pd.read_csv('../datasets/chess/lichess_db_puzzle.csv.zst')
    data = [(update_board(csv['FEN'][i], csv['Moves'][i][:4]), ' '.join(csv['Moves'][i].split(' ')[1:])[:num_moves - 1]) for
            i in range(len(csv))
            if 'mateIn' + str(2) in csv['Themes'][i]]

    chess_dict = {}
    chess_dict['prefix'] = [data[i][0] for i in range(350000)]
    chess_dict['target'] = [data[i][1] for i in range(350000)]
    with open('../datasets/chess/mate_in_2_train.json', 'w') as f:
        json.dump(chess_dict, f)

    chess_dict = {}
    chess_dict['prefix'] = [data[i][0] for i in range(350000, len(data))]
    chess_dict['target'] = [data[i][1] for i in range(350000, len(data))]
    with open('../datasets/chess/mate_in_2_test.json', 'w') as f:
        json.dump(chess_dict, f)"""
    from tokenizing import get_tokenizer
    import types
    args = types.SimpleNamespace()
    args.model = 'gpt2'
    tokenizer = get_tokenizer(args)
    data = Chess(tokenizer, 5000, '../data/datasets/chess/mate_in_2_test.json', device=device, unrolled=True)



