"""Torch Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

import dgl.function as fn
from dgl.utils import expand_as_pair


class GINConv(nn.Module):

    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

        #self.lin_e = nn.Linear(64, 64) ## edge features
        #self.eps_efeat = th.nn.Parameter(th.FloatTensor([init_eps]))


    def forward(self, graph, n_feat, e_feat, edge_weight):

        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                assert e_feat.shape[0] == graph.number_of_edges()
                graph.edata['_edge_h'] = edge_weight
                #graph.edata['_edge_weight'] = edge_weight
                #graph.edata['_edge_features'] = e_feat#edge_weight.view(-1,1) * e_feat
                #graph.apply_edges(lambda edges: {'_edge_h': edges.data['_edge_weight'].view(-1,1) * edges.data['_edge_features']})
                aggregate_fn = fn.u_mul_e('h', '_edge_h', 'm')

            feat_src, feat_dst = expand_as_pair(n_feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst