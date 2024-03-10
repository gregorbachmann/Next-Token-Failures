import torch
from torch.utils.data import Dataset
import numpy as np
import random


def star_graph(degSource, pathLen, numNodes, reverse=False):
    source = np.random.randint(0, numNodes, 1)[0]
    goal = np.random.randint(0, numNodes, 1)[0]
    while goal == source:
        goal = np.random.randint(0, numNodes, 1)[0]

    path = [source]
    edge_list = []

    # Choose random nodes along the path
    for _ in range(pathLen - 2):
        node = np.random.randint(0, numNodes, 1)[0]
        while node in path or node == goal:
            node = np.random.randint(0, numNodes, 1)[0]
        path.append(node)

    path.append(goal)
    # Connect the path
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    remaining_nodes = []
    for i in range(numNodes):
        if i not in path:
            remaining_nodes.append(i)

    i = 0
    deg_nodes = set()
    while i < degSource - 1:
        # Add neighbour to source
        node = source
        next_node = np.random.randint(0, numNodes, 1)[0]
        l = 1
        while l < pathLen:
            if next_node not in deg_nodes and next_node not in path:
                edge_list.append([node, next_node])
                deg_nodes.add(next_node)
                node = next_node
                l += 1
            next_node = np.random.randint(0, numNodes, 1)[0]

        i += 1

    random.shuffle(edge_list)
    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal


def generate_and_save(n_train, n_test, degSource, pathLen, numNodes, reverse=False):
    """
    Generate a list of train and testing graphs and save them for reproducibility
    """
    file = open('../data/datasets/graphs/' + 'deg_' + str(degSource) + '_path_' + str(pathLen) + '_nodes_' + str(
        numNodes) + '_train_' +
          str(n_train) + '.txt', 'w')

    for i in range(n_train):
        path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
        path_str = ''
        for node in path:
            path_str += str(node) + ','
        path_str = path_str[:-1]

        edge_str = ''
        for e in edge_list:
            edge_str += str(e[0]) + ',' + str(e[1]) + '|'
        edge_str = edge_str[:-1]
        edge_str += '/' + str(start) + ',' + str(goal) + '='

        out = edge_str + path_str
        file.write(out + '\n')
    file.close()

    file = open('../data/datasets/graphs/' + 'deg_' + str(degSource) + '_path_' + str(pathLen) + '_nodes_' +
                str(numNodes) + '_test_' +
                 str(n_test) + '.txt', 'w')

    for i in range(n_test):
        path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
        path_str = ''
        for node in path:
            path_str += str(node) + ','
        path_str = path_str[:-1]

        edge_str = ''
        for e in edge_list:
            edge_str += str(e[0]) + ',' + str(e[1]) + '|'
        edge_str = edge_str[:-1]
        edge_str += '/' + str(start) + ',' + str(goal) + '='

        out = edge_str + path_str
        file.write(out + '\n')

    file.close()


def prefix_target_list(filename=None, reverse=False):
    """
    Load graphs and split them into prefix and target and return the list
    """
    data_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        prefix = line.strip().split('=')[0] + '='
        target = line.strip().split('=')[1]
        if reverse:
            target = ','.join(target.split(',')[::-1])
        data_list.append((prefix, target))

    return data_list


class Graphs(Dataset):
    def __init__(self, tokenizer, n_samples, data_path, device, eval=False, teacherless_token=None, reverse=False):
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.eval_mode = eval
        self.data_path = data_path
        self.teacherless_token = teacherless_token
        self.reverse = reverse

        self.data_file = prefix_target_list(self.data_path, reverse=reverse)[:n_samples]
        self.tokenized, self.num_prefix_tokens, self.num_target_tokens = tokenizer.tokenize(self.data_file)

        self.num_tokens = self.num_prefix_tokens + self.num_target_tokens

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs
        x = self.tokenized[idx][:-1].clone()
        if self.teacherless_token is not None:
            x[self.num_prefix_tokens:] = self.teacherless_token
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


def get_edge_list(x, num_nodes, path_len):
    """
    Given the tokenised input for the Transformer, map back to the edge_list
    """
    edge_list = []
    pair = []
    x = x.squeeze().cpu().numpy()

    for i, n in enumerate(x):
        if n in range(num_nodes):
            pair.append(n)
        if len(pair) == 2:
            edge_list.append(pair)
            pair = []
        if n == num_nodes + 2:
            break

    start = x[i + 1]
    goal = x[i + 2]
    path = [x[i + j] for j in range(4, 4 + path_len)]

    return edge_list, start, goal, path


def get_edge_list_byte(x, num_nodes, path_len, decode):
    """
    Given the tokenised input for the Transformer, map back to the edge_list
    """
    edge_list = []
    x = list(x.squeeze().cpu().numpy())
    dec = [decode([val]) for val in x]
    edge = []
    for i, val in enumerate(dec):
        if val not in [',', '|', '=', '->']:
            edge.append(val)
        if len(edge) == 2:
            edge_list.append(edge)
            edge = []

        if val == '->':
            break
    i += 2
    start = dec[i + 1]
    goal = dec[i -1]
    path = [dec[i + 3 + 2 * j] for j in range(0, path_len - 2)]

    return edge_list, start, goal, path


if __name__ == '__main__':
    import types
    from data import get_dataset
    from tokenizing import get_tokenizer

    # Create graphs and save
    n_train = 200000
    n_test = 20000
    deg = 2
    path_len = 5
    num_nodes = 50
    reverse = False
    generate_and_save(n_train=n_train, n_test=n_test, degSource=deg, pathLen=path_len, numNodes=num_nodes,
                      reverse=reverse)

    # Load data
    device = 'cpu'
    args = types.SimpleNamespace(model='gpt', num_nodes=num_nodes)
    args.dataset = 'graph'
    args.deg = deg
    args.path_len = path_len
    args.n_train = n_train
    args.n_test = n_test
    args.reverse = False

    args.dollar = 11
    tokenizer = get_tokenizer(args)
    trainset, testset = get_dataset(args, tokenizer, device)
    print(trainset.num_tokens)
    trainset.__getitem__(10)
    trainset.eval()
    trainset.__getitem__(10)

    import matplotlib.pyplot as plt
    import networkx as nx
    path, edge_list, start, goal = star_graph(deg, path_len, num_nodes, reverse=reverse)
    print(len(edge_list))
    print(path)
    print(edge_list)
    print('Start:', start, 'Goal:', goal)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    nx.draw(G, with_labels=True)
    plt.show()