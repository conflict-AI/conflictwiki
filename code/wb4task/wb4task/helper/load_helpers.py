import torch


def trans_get_splits(g, n_val=0.33, n_test=0.33):
    ## splits graph by nodes

    edge_id_list = g.edges(form ="eid", order="eid")

    n_train = int((1- (n_val + n_test)) * len(edge_id_list))
    n_val =  int(n_val * len(edge_id_list))

    random_edge_id_list = torch.randperm(len(edge_id_list))

    train_edges = random_edge_id_list[:n_train]
    val_edges = random_edge_id_list[n_train: (n_train + n_val)]
    test_edges = random_edge_id_list[(n_train + n_val):]

    mask = torch.zeros(len(edge_id_list), dtype=torch.bool)
    mask[train_edges] = True
    g.edata["train_mask"] = mask

    mask = torch.zeros(len(edge_id_list), dtype=torch.bool)
    mask[val_edges] = True
    g.edata["val_mask"] = mask

    mask = torch.zeros(len(edge_id_list), dtype=torch.bool)
    mask[test_edges] = True
    g.edata["test_mask"] = mask

    return train_edges, val_edges, test_edges



def randomise_node_features(g, random_node_frac=0.5, random_node_type="mean", return_n = None):

    ## choose nodes
    num_random_nodes = int(random_node_frac * g.number_of_nodes())

    if return_n != None: ## simply return 2 random nodes from whole graph
        random_nodes = torch.randperm(g.number_of_nodes())[:return_n]
        return g.ndata["node_features"][random_nodes]

    random_nodes = torch.randperm(g.number_of_nodes())[:num_random_nodes]

    ## construct random node features
    if random_node_type == "mean":
        random_node_features = torch.mean(g.ndata["node_features"], dim=0).view(1, -1)

    ## assign random node features
    g.ndata["node_features"][random_nodes] = random_node_features




def randomise_labels(g, random_label_frac=0.5):

    ## choose edges
    num_random_labels = int(random_label_frac * g.number_of_edges())
    random_edges = torch.randperm(g.number_of_edges())[:num_random_labels]

    ## construct random edge features
    random_label_mask = torch.zeros((g.number_of_edges()))
    random_label_mask[random_edges] = True

    ## assign random edge features
    g.edata["label_mask"] = random_label_mask

