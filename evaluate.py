import torch
from tqdm import tqdm

from utils.training_utils import AverageMeter


# Function to evaluate performance when generating
@torch.no_grad()
def evaluate(model, loader, ctx, temperature, top_k, results=None, mode='test'):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens

    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    total_acc = AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    #model.set_cache(loader.dataset.device)
    for x in bar:
        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()

        with ctx:
            y_pred = model.generate(x, num_target_tokens, temperature=temperature, top_k=top_k)
        #model.reset_cache()

        # Check how many tokens we get right and how many predictions are completely correct
        correct = y.eq(y_pred[:, -num_target_tokens:]).float()

        # Completely correct
        completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
        total_acc.update(completely_correct.item(), x.shape[0])

        # Individual token accuracy
        per_token_acc = correct.mean(dim=0)
        for i in range(num_target_tokens):
            tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])

        bar.set_description(f'{mode} accuracy: {total_acc.get(percentage=True):.2f}')

    #model.empty_cache()

    # Switch back to train mode
    loader.dataset.train()
    model.train()

    if results is not None:
        results[mode + '/accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
    return results


# Function to evaluate performance when applying teacher forcing
@torch.no_grad()
def evaluate_forced(model, loader, ctx, results=None, mode='test'):
    """
    Generates sequences with teacher-forcing and calculates accuracies
    """
    num_target_tokens = loader.dataset.num_target_tokens
    total_acc, total_loss = AverageMeter(), AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    for x, y in bar:
        # Produce logits with teacher-forcing (i.e. like during training)
        with ctx:
            logits, loss, accs = model(x, y)

        total_acc.update(val=accs['acc'], num=x.shape[0])
        total_loss.update(val=loss, num=x.shape[0])
        for i in range(num_target_tokens):
            tokens_corr[i].update(accs['token_acc'], x.shape[0])

        bar.set_description('Forced Loss: {:.4f} Forced Acc: {:.2f}'.format(total_loss.get(),
                                                              total_acc.get(percentage=True)))

    if results is not None:
        results[mode + '/forced loss'] = total_loss.get()
        results[mode + '/forced accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)

    return results
