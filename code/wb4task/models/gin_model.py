
import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl import nn as dglnn
import dgl.function as fn
import torch as th
from dgl.nn.pytorch.conv import NNConv

from wb4task.helper.train_helpers import get_edge_weight, update_nodes_with_edges, get_edge_features, get_node_features, pass_edge_features
from wb4task.models import ginConv



class SignedGIN(nn.Module):

    def __init__(self, neighborhood_steps, in_features, hidden_features, out_features):
        super().__init__()

        lin_0 = nn.Linear(in_features, hidden_features)
        self.conv0 = ginConv.GINConv(apply_func = lin_0, aggregator_type = 'mean')
        #self.conv0 = dglnn.GINConv(apply_func=lin_0, aggregator_type='mean')
        lin_1 = nn.Linear(hidden_features, out_features)
        self.conv1 = ginConv.GINConv(apply_func = lin_1, aggregator_type = 'mean')
        #self.conv1 = dglnn.GINConv(apply_func=lin_1, aggregator_type='mean')
        lin_2 = nn.Linear(hidden_features, out_features)
        self.conv2 = ginConv.GINConv(apply_func = lin_2, aggregator_type = 'mean')
        #self.conv2 = dglnn.GINConv(apply_func=lin_2, aggregator_type='mean')

        self.edge_norm = dglnn.EdgeWeightNorm(norm='right')
        self.neighborhood_steps = neighborhood_steps



    def forward(self, node_feature_reducer, edge_feature_reducer, blocks):
        #https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=graphconv#dgl.nn.pytorch.conv.GraphConv

        n = node_feature_reducer(blocks[0].srcdata['node_features']) ## input node features

        if self.neighborhood_steps >= 1:
            norm_edge_weight = get_edge_weight(blocks[0], self.edge_norm)
            e = edge_feature_reducer(blocks[0].edata['edge_features'])
            h = F.relu(self.conv0(blocks[0], n, e, edge_weight=norm_edge_weight))
            #h = F.relu(self.conv0(blocks[0], n, edge_weight=norm_edge_weight))

        if self.neighborhood_steps >= 2:
            norm_edge_weight = get_edge_weight(blocks[1], self.edge_norm)
            e = edge_feature_reducer(blocks[1].edata['edge_features'])
            h = F.relu(self.conv0(blocks[1], h, e, edge_weight=norm_edge_weight))
            #h = F.relu(self.conv1(blocks[1], h, edge_weight=norm_edge_weight))

        if self.neighborhood_steps >= 3:
            norm_edge_weight = get_edge_weight(blocks[2], self.edge_norm)
            e = edge_feature_reducer(blocks[2].edata['edge_features'])
            h = F.relu(self.conv2(blocks[2], h, e, edge_weight=norm_edge_weight))
            #h = F.relu(self.conv2(blocks[2], h, edge_weight=norm_edge_weight))
        return h

