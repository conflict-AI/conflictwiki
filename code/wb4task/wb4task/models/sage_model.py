
from torch import nn
from torch.nn import functional as F
from dgl import nn as dglnn

from wb4task.helper.train_helpers import get_edge_weight, pass_edge_features


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool', norm=lambda x: F.normalize(x))
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, node_features, edge_features):
        # inputs are features of nodes

        norm_edge_weight = get_edge_weight(graph)
        graph, node_h_features, edge_h_features = pass_edge_features(graph, node_features, edge_features)
        h = F.relu(self.conv1(graph, node_h_features, edge_weight=norm_edge_weight))

        norm_edge_weight = get_edge_weight(graph)
        graph, node_h_features, edge_h_features = pass_edge_features(graph, h, edge_features)
        h = F.relu(self.conv2(graph, node_h_features, edge_weight=norm_edge_weight))
        return h





