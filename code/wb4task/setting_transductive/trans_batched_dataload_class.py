from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb4task.task_construction.nx_to_dgl import construct_networkx
from wb4task.helper.load_helpers import trans_get_splits, randomise_labels, randomise_node_features, randomise_edge_features

import torch
import numpy as np
import dgl
import itertools


class WikiNetworkData():
    def __init__(self, config):

        nx_graph = construct_networkx(config, undirected=False)
        g_directed = dgl.from_networkx(nx_graph, node_attrs=["node_features"],edge_attrs=["edge_features", "label_discrete"])  # edge_id_attr_name = "edge_id",
        self.g, self.edge_ids, self.reverse_edge_ids = self.add_reverse_edges(g_directed)


    def add_reverse_edges(self,g):

        edge_ids = g.edges(form='all', order='eid')[2]
        n_edge = len(edge_ids)

        reverse_edge_ids = edge_ids + n_edge

        g_reverse = g.reverse(copy_ndata=True, copy_edata=True)
        g.add_edges(g_reverse.edges()[0], g_reverse.edges()[1], g_reverse.edata)

        return g, edge_ids, reverse_edge_ids


    def get_class_weights(self):

        class_labels, class_counts = np.unique(self.g.edata['label_discrete'], return_counts=True)
        print("target labels:", class_labels, class_counts)

        self.class_labels, self.class_counts = class_labels[::-1], class_counts[::-1]
        self.class_weights = torch.tensor(self.class_counts.copy(), dtype=torch.float) / self.g.number_of_edges()

        return self.class_labels, self.class_counts, self.class_weights



    def edge_data_loader(self, sample_edge_ids, batch_size = 32, neighborhood_steps = 2):

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=neighborhood_steps, return_eids=True)
        #sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 15], replace =False, return_eids=True)
        reverse_sample_edge_ids = self.reverse_edge_ids[sample_edge_ids]

        #sample_edge_ids = torch.cat((self.edge_ids, self.reverse_edge_ids), dim = 0)
        #reverse_sample_edge_ids = torch.cat((self.reverse_edge_ids, self.edge_ids), dim = 0)

        g_sample = self.g.remove_edges(sample_edge_ids)

        dataloader = dgl.dataloading.EdgeDataLoader(
            self.g, sample_edge_ids, sampler,
            batch_size=batch_size,
            g_sampling = g_sample,
            shuffle=True,
            drop_last=True,
            exclude = None,
            reverse_eids = reverse_sample_edge_ids,
            num_workers=0)

        return dataloader


def load_data(config, batch_size=32, n_val=0.2, n_test=0.1, neighborhood_steps=2, random_node_frac = 0.0, random_edge_frac = 0.0, random_label_frac = 0.0, random_seed = 0):
    torch.manual_seed(random_seed)
    wikinetworkdata = WikiNetworkData(config)
    class_labels, class_counts, class_weights = wikinetworkdata.get_class_weights()
    label_info = {"class_labels": class_labels, "class_counts": class_counts, "class_weights": class_weights}

    train_edges, val_edges, test_edges = trans_get_splits(wikinetworkdata, n_val=n_val, n_test=n_test)

    train_dl = wikinetworkdata.edge_data_loader(train_edges, batch_size=batch_size, neighborhood_steps=neighborhood_steps)
    val_dl = wikinetworkdata.edge_data_loader(val_edges, batch_size=len(val_edges), neighborhood_steps=neighborhood_steps)
    test_dl = wikinetworkdata.edge_data_loader(test_edges, batch_size=len(test_edges), neighborhood_steps=neighborhood_steps)

    #train_dl = wikinetworkdata.edge_data_loader(train_edges, batch_size=1, neighborhood_steps=neighborhood_steps)
    #val_dl = wikinetworkdata.edge_data_loader(val_edges, batch_size=2, neighborhood_steps=neighborhood_steps)
    #test_dl = wikinetworkdata.edge_data_loader(test_edges, batch_size=2, neighborhood_steps=neighborhood_steps)

    randomise_node_features(wikinetworkdata.g, random_node_frac=random_node_frac, random_node_type="mean")
    randomise_edge_features(wikinetworkdata.g, random_edge_frac=random_edge_frac, random_edge_type="mean")
    randomise_labels(wikinetworkdata.g, random_label_frac=random_label_frac)

    print("batch size:", batch_size, "train edges:", len(train_edges), ", val edges:", len(val_edges), ", test edges:", len(test_edges))
    return train_dl, val_dl, test_dl, wikinetworkdata.g, label_info, wikinetworkdata



if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size=500, n_val=0.3, n_test=0.1, neighborhood_steps=2, random_node_frac = 1.0, random_edge_frac = 1.0, random_label_frac = 1.0)

    for i, (input_nodes, edge_subgraph, blocks) in enumerate(val_dl):
        print(blocks)
        print(blocks[1].dstdata['_ID'])
        print(edge_subgraph.edata['_ID'])
