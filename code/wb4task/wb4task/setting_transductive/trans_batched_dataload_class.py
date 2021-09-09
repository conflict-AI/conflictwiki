from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb4task.task_construction.nx_to_dgl import construct_networkx
from wb4task.helper.load_helpers import trans_get_splits, randomise_labels, randomise_node_features

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



    def edge_data_loader(self, batch_size = 32, neighborhood_steps = 2):

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=neighborhood_steps, return_eids=True)

        sample_edge_ids = self.edge_ids
        reverse_sample_edge_ids = self.reverse_edge_ids

        #sample_edge_ids = torch.cat((self.edge_ids, self.reverse_edge_ids), dim = 0)
        #reverse_sample_edge_ids = torch.cat((self.reverse_edge_ids, self.edge_ids), dim = 0)

        dataloader = dgl.dataloading.EdgeDataLoader(
            self.g, sample_edge_ids, sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            exclude = None,
            reverse_eids = reverse_sample_edge_ids,
            num_workers=0)

        return dataloader


def load_data(config, batch_size=32, n_val=0.2, n_test=0.05, neighborhood_steps=2, random_node_frac = 0.0, random_label_frac = 0.0):

    wikinetworkdata = WikiNetworkData(config)
    class_labels, class_counts, class_weights = wikinetworkdata.get_class_weights()
    label_info = {"class_labels": class_labels, "class_counts": class_counts, "class_weights": class_weights}
    dl = wikinetworkdata.edge_data_loader(batch_size=batch_size, neighborhood_steps=neighborhood_steps)

    val_test_dl = wikinetworkdata.edge_data_loader(batch_size=len(wikinetworkdata.g.edges()[0]), neighborhood_steps=neighborhood_steps)

    randomise_node_features(wikinetworkdata.g, random_node_frac=random_node_frac, random_node_type="mean")
    randomise_labels(wikinetworkdata.g, random_label_frac=random_label_frac)

    train_edges, val_edges, test_edges = trans_get_splits(wikinetworkdata.g, n_val=n_val, n_test=n_test)
    print("batch size:", batch_size, "train edges:", len(train_edges), ", val edges:", len(val_edges), ", test edges:", len(test_edges))

    return dl, val_test_dl, wikinetworkdata.g, label_info, wikinetworkdata



if __name__ == "__main__":

    config = configs.ConfigBase()
    dl, val_test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size=32, n_val=0.1, n_test=0.05, neighborhood_steps=2, random_node_frac = 0.0, random_label_frac = 1.0)
