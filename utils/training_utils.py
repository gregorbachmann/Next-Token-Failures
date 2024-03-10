import math
import torch


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.val = 0

    def update(self, val, num):
        self.val += val * num
        self.num += num

    def get(self, percentage=False):
        val = self.val / self.num * 100 if percentage else self.val / self.num
        return val


def accuracy(logits, targets):
    num_prefix_tokens = targets[0].eq(-1).sum().item()
    num_target_tokens = targets.shape[1] - num_prefix_tokens
    targets = targets[:, num_prefix_tokens:]
    logits = logits[:, num_prefix_tokens:, :]
    correct = torch.argmax(logits, dim=-1).eq(targets).to(torch.float)
    seq_correct = torch.sum(correct, dim=1).eq(num_target_tokens).float()
    acc = torch.mean(seq_correct)
    per_token_acc = correct.mean(dim=0)

    return acc, per_token_acc


def get_run_name(args):
    name = args.dataset
    if args.dataset == 'graph':
        name += '_deg' + str(args.deg) + '_path_' + str(args.path_len) + 'num_nodes_' + str(args.num_nodes) + \
                '_ntrain_' + str(args.n_train) + '_teacherless_' + str(args.teacherless) + '_reverse_' + str(args.reverse)
    elif args.dataset == 'chess':
        name += '_mate_in_' + str(args.mate_in) + '_ntrain_' + str(args.n_train) + '_unrolled_' + str(args.unrolled) + \
                '_teacherless_' + str(args.teacherless)

    return name
