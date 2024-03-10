import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

from tokenizing import get_tokenizer
from utils.training_utils import get_lr, get_run_name, AverageMeter
from data import get_dataset
from evaluate import evaluate, evaluate_forced
from models import get_model


# Parse arguments
parser = argparse.ArgumentParser(description="Next-token prediction")
# Data
parser.add_argument(
        "--model", type=str, default='gpt', help="Learning rate",
    )
parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers",
    )
parser.add_argument(
        "--n_embd", type=int, default=240, help="Embedding size",
    )
parser.add_argument(
        "--n_head", type=int, default=6, help="Number of heads",
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=10000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=5000, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=5000, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb username",
    )

args = parser.parse_args()
# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = 1

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Optimiser
dtype = 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = 1e-5

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

max_iters = len(train_data) * args.epochs
lr_decay_iters = max_iters

args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None
model = get_model(args)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Setup wandb logging
if wandb_log:
    wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

results = {}
num_iters = 0

for ep in range(args.epochs):
    if ep % args.save_every == 0 and ep > 0:
        torch.save(model.state_dict(), path + "_epoch_" + str(ep))

    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()

    for x, y in train_bar:
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            logits, loss, accs = model(x, y)

        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(accs['acc'], x.shape[0])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'.format(ep, args.epochs, total_loss.get(),
             total_acc.get(percentage=True))
        )

    # evaluate the loss on train/val sets and write checkpoints
    if ep % args.eval_every == 0:
        # Generate sequences and check accuracies
        if args.eval_train:
            results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='train')
            results = evaluate_forced(model, train_loader, results=results, mode='train')

        results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='test')
        results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='test')

        if wandb_log:
            wandb.log(results)
