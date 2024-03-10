# Next-Token-Failures

![](https://github.com/gregorbachmann/Next-Token-Failures/blob/main/imgs/cleverhans.png)

This is the code used to produce the results presented in the paper TO BE INSERTED.

## Requirements
1. *torch 2.2.0*
2. *transformers 4.37.2*
3. *numpy 1.26.3*
4. *tqdm 4.66.1*
5. *wandb 4.66.1*


## Usage
In order to train a GPT-style model from scratch on a star graph with degree 2 and path length 5 with 50 possible node values, run
> python3 finetune.py --n_train 200000 --n_test 20000 --model gpt --batch_size 256 --dataset graph --deg 2 --path 5 --lr 0.0001
>
