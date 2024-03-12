# Next-Token-Failures

![](https://github.com/gregorbachmann/Next-Token-Failures/blob/main/imgs/cleverhans.png)

This is the code used to produce the results presented in the paper <https://arxiv.org/abs/2403.06963>.

## Requirements
The following packages are needed to run the code:
1. *torch* 2.2.0
2. *transformers* 4.37.2
3. *numpy* 1.26.3
4. *tqdm* 4.66.1
5. *wandb* 0.16.2


## Usage
In order to train a GPT-style model from scratch with standard next-token prediction on a star graph with degree 2 and path length 5 with 50 possible node values, run the command
> python3 train.py --model gpt --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001

To train the same model using the reverse encoding, add the flag *--reverse*. In order to train with our teacherless objective, add the flag --teacherless. 

To finetune a pre-trained model like GPT2-large, run the command
>python3 finetune.py --model gpt2-large --n_train 200000 --n_test 20000  --batch_size 16 --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.00001
>
Similarly, you can finetune a Pythia model using the flag --model pythia-410m-deduped. You can also add the flags for reversing and teacherless training as outlined above.
