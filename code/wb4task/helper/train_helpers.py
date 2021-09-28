import torch
import dgl
import dgl.function as fn

def get_edge_weight(graph, edge_norm):

    edge_weight = graph.edata['label_discrete'].float()  ## get edge signs
    edge_weight[edge_weight == 1.0] = 1.0  ## friends ## 1.0
    edge_weight[edge_weight == 0.0] = -1.0 ## enemies ## -1.0

    label_mask = graph.edata['label_mask']#.float()  ## get edge signs
    edge_weight[label_mask == True] = (torch.randint(0, 2, (1,)).item() * 2) - 1 ## mask edges ## 1.0 / -1.0 at random

    #norm_edge_weight= edge_norm(graph, edge_weight)
    return edge_weight


def edata_message_func(edges):
    node_edge_mean = torch.stack((edges.src['node_h_features'], edges.data['edge_h_features']),2)
    node_edge_mean = torch.mean(node_edge_mean, dim = 2)
    return {'node_edge_mean': node_edge_mean}
    #return {'node_edge_cat': ((edges.src['node_features'] + edges.data['edge_features']) / 2)}
    #return {'node_edge_cat': torch.cat((edges.src['node_features'], edges.data['edge_features']), 1)}



def update_nodes_with_edges(graph, node_features, edge_features):
    #https://docs.dgl.ai/en/0.6.x/guide/message-api.html

    graph.srcdata['node_features'] = node_features
    graph.edata['edge_features'] = edge_features

    graph.update_all(fn.u_mul_e('node_features', 'edge_features', 'm'), fn.mean('m', 'node_features'))
    dst_node_features = graph.dstdata['node_features']
    return dst_node_features


def pass_edge_features(graph, node_features, edge_features):

    graph.ndata["node_h_features"] = node_features
    graph.edata["edge_h_features"] = edge_features

    graph.update_all(edata_message_func, dgl.function.mean('node_edge_mean', 'node_h_features'))

    node_features = get_node_features(graph, 'node_h_features')
    edge_features = get_edge_features(graph, 'edge_h_features')

    return graph, node_features, edge_features


def get_edge_features(graph, feature_name):
    edge_features = graph.edata[feature_name].float()  ## get edge features
    return edge_features


def get_node_features(graph, feature_name):
    node_features = graph.ndata[feature_name].float()  ## get edge features
    return node_features

