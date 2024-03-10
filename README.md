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
In order to train a GPT-style model from scratch with standard next-token prediction on a star graph with degree 2 and path length 5 with 50 possible node values, run
> python3 train.py --model gpt --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001

To train the same model using the reverse encoding, add the flag *--reverse*. In order to train with our teacherless objective, add the flag --teacherless. 

To finetune a pre-trained model like GPT2-large, run 
>python3 finetune.py --model gpt2-large --n_train 200000 --n_test 20000  --batch_size 16 --dataset graph --deg 2 --path 5 --num_nodes 50 --eval_every 10000 --lr 0.00001
Similarly, you can finetune a Pythia model using the flag --model pythia-410m-deduped.
