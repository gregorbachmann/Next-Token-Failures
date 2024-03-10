numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder = {str(i): i for i in range(num_nodes)}
        self.encoder['|'] = num_nodes
        self.encoder['='] = num_nodes + 1
        self.encoder['/'] = num_nodes + 2
        self.encoder['$'] = num_nodes + 3

        self.decoder = {i: i for i in range(num_nodes)}
        self.decoder[num_nodes] = '|'
        self.decoder[num_nodes + 1] = '='
        self.decoder[num_nodes + 2] = '/'
        self.decoder[num_nodes + 3] = '$'
        self.decoder[-1] = ':'

    def encode(self, x):
        out = []
        i = 0
        while i < len(x):
            if x[i] == ',':
                i += 1
                continue
            s = ''
            j = 0
            while i + j < len(x) and x[i + j] in numbers:
                s += x[i + j]
                j += 1
            if s == '':
                s = x[i]
                i += 1
            else:
                i += j
            out.append(self.encoder[s])

        return out

    def decode(self, x):
        return [self.decoder[i] for i in x]
