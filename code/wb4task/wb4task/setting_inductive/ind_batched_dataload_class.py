from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import torch
import numpy as np
import networkx as nx
import dgl
import itertools

from wb4task.task_construction.nx_to_dgl import construct_networkx
from wb4task.helper.load_helpers import randomise_labels, randomise_node_features


class WikiNetworkData():
    def __init__(self, config):

        nx_graph = construct_networkx(config, undirected=False)
        g_directed = dgl.from_networkx(nx_graph, node_attrs=["node_features"],edge_attrs=["edge_features", "label_discrete"])  # edge_id_attr_name = "edge_id",
        self.g, self.n_edge = self.add_reverse_edges(g_directed)


    def add_reverse_edges(self, g):
        n_edge = len(g.edges(form='all', order='eid')[2])
        g_reverse = g.reverse(copy_ndata=True, copy_edata=True)
        g.add_edges(g_reverse.edges()[0], g_reverse.edges()[1], g_reverse.edata)
        return g, n_edge


    def get_splits(self, n_val=0.33, n_test=0.33):
        ## splits graph by nodes
        train_g, val_g, test_g = dgl.data.utils.split_dataset(self.g, frac_list=[1 - (n_val + n_test), n_val, n_test], shuffle=True, random_state=2)
        return train_g, val_g, test_g


    def get_class_weights(self):
        class_labels, class_counts = np.unique(self.g.edata['label_discrete'], return_counts=True)
        print("target labels:", class_labels, class_counts)

        self.class_labels, self.class_counts = class_labels[::-1], class_counts[::-1]
        self.class_weights = torch.tensor(self.class_counts.copy(), dtype=torch.float) / self.g.number_of_edges()

        return self.class_labels, self.class_counts, self.class_weights


    def get_reverse_edge_id(self, edge_id):
        reverse_edge_id = edge_id + self.n_edge
        return reverse_edge_id


    def get_edge_ids(self, node_id_list):
        node_permute = list(itertools.permutations(node_id_list, r=2))
        edge_u = [edge[0] for edge in node_permute]
        edge_v = [edge[1] for edge in node_permute]
        edge_list = self.g.edge_ids(edge_u, edge_v, return_uv=True)
        edge_id = edge_list[2]
        reverse_edge_id = self.get_reverse_edge_id(edge_id)
        return edge_id, reverse_edge_id


    def edge_data_loader(self, g_dataset, batch_size = 32, neighborhood_steps = 2):

        edge_id, reverse_edge_id = self.get_edge_ids(g_dataset.indices)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers=neighborhood_steps, return_eids=True)

        dataloader = dgl.dataloading.EdgeDataLoader(
            self.g, edge_id, sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            reverse_eids = reverse_edge_id,
            num_workers=1)

        return dataloader



def load_data(config, batch_size=32, n_val=0.2, n_test=0.05, neighborhood_steps=2,  random_node_frac = 0.0, random_label_frac = 0.0):

    wikinetworkdata = WikiNetworkData(config)
    class_labels, class_counts, class_weights = wikinetworkdata.get_class_weights()
    label_info = {"class_labels": class_labels, "class_counts": class_counts, "class_weights": class_weights}

    train_dataset, val_dataset, test_dataset = wikinetworkdata.get_splits(n_val=n_val, n_test=n_test)
    print("batch size:", batch_size, ", train nodes:", len(train_dataset.indices), ", val nodes:", len(val_dataset.indices), ", test nodes:", len(test_dataset.indices))

    randomise_node_features(wikinetworkdata.g, random_node_frac=random_node_frac, random_node_type="mean")
    randomise_labels(wikinetworkdata.g, random_label_frac=random_label_frac)

    train_dl = wikinetworkdata.edge_data_loader(train_dataset, batch_size=batch_size, neighborhood_steps=neighborhood_steps)
    val_dl = wikinetworkdata.edge_data_loader(val_dataset, batch_size=val_dataset.dataset.num_nodes(), neighborhood_steps=neighborhood_steps)
    test_dl = wikinetworkdata.edge_data_loader(test_dataset, batch_size=test_dataset.dataset.num_nodes(), neighborhood_steps=neighborhood_steps)

    print("batches – train_dl:", len(train_dl), "val_dl:", len(val_dl), "test_dl:", len(test_dl), "\n")
    return train_dl, val_dl, test_dl, label_info, wikinetworkdata



if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, label_info, wikinetworkdata = load_data(config, batch_size=32, n_val=0.1, n_test=0.05, neighborhood_steps=2)
