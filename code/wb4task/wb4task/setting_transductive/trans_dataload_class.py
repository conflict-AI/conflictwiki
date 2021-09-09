from wb0configs import configs
from wb4task.helper.load_helpers import trans_get_splits, randomise_labels, randomise_node_features
from wb4task.task_construction.nx_to_dgl import construct_networkx

import torch
import numpy as np
import dgl


class WikiNetworkData():
    def __init__(self, config):

        nx_graph = construct_networkx(config, undirected=True)
        self.g = dgl.from_networkx(nx_graph, node_attrs=["node_features"],edge_attrs=["edge_features", "label_discrete"])  # edge_id_attr_name = "edge_id",


    def mask_fraction(self, edge_id_list, mask_out_frac = 0.2, mask_type = 'train_mask'):
        self.g.edata[mask_type] = torch.zeros(len(edge_id_list), dtype=torch.bool).bernoulli(mask_out_frac)


    def get_class_weights(self):
        class_labels, class_counts = np.unique(self.g.edata['label_discrete'], return_counts=True)
        print("target labels:", class_labels, class_counts)

        self.class_labels, self.class_counts = class_labels, class_counts #class_labels[::-1], class_counts[::-1]
        self.class_weights = torch.tensor(self.class_counts.copy(), dtype=torch.float) / self.g.number_of_edges()

        return self.class_labels, self.class_counts, self.class_weights



def load_data(config, n_val=0.2, n_test=0.05, random_node_frac = 0.0, random_label_frac = 0.0):

    wikinetworkdata = WikiNetworkData(config)
    class_labels, class_counts, class_weights = wikinetworkdata.get_class_weights()
    label_info = {"class_labels": class_labels, "class_counts": class_counts, "class_weights": class_weights}

    randomise_node_features(wikinetworkdata.g, random_node_frac=random_node_frac, random_node_type="mean")
    randomise_labels(wikinetworkdata.g, random_label_frac=random_label_frac)

    train_edges, val_edges, test_edges = trans_get_splits(wikinetworkdata.g, n_val=n_val, n_test=n_test)
    print("train edges:", len(train_edges), ", val edges:", len(val_edges), ", test edges:", len(test_edges))



    return wikinetworkdata.g, label_info, wikinetworkdata



if __name__ == "__main__":

    config = configs.ConfigBase()
    g, label_info, wikinetworkdata = load_data(config, n_val=0.1, n_test=0.1, random_node_frac = 0.0, random_label_frac = 0.0)

