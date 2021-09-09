
import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl import nn as dglnn
import dgl.function as fn
import torch as th

from wb4task.helper.train_helpers import get_edge_weight, get_edge_features, get_node_features, pass_edge_features

class SignedGCN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        lin_1 = nn.Linear(in_features, hidden_features)
        self.conv1 = dglnn.GINConv(apply_func = lin_1, aggregator_type = 'mean')
        lin_2 = nn.Linear(hidden_features, out_features)
        self.conv2 = dglnn.GINConv(apply_func = lin_2, aggregator_type = 'mean')
        self.edge_norm = dglnn.EdgeWeightNorm(norm='right')


    def forward(self, edge_subgraph, blocks, n, e):
        #https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=graphconv#dgl.nn.pytorch.conv.GraphConv

        norm_edge_weight = get_edge_weight(blocks[0], self.edge_norm)
        h = F.relu(self.conv1(blocks[0], n, edge_weight=norm_edge_weight))

        norm_edge_weight = get_edge_weight(blocks[1], self.edge_norm)
        x = F.relu(self.conv2(blocks[1], h, edge_weight=norm_edge_weight))

        return x

