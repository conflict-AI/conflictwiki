import torch


def trans_get_splits(wikinetworkdata, n_val=0.33, n_test=0.33):
    ## splits graph by nodes
    #edge_id_list = g.edges(form ="eid", order="eid")

    g = wikinetworkdata.g
    edge_id_list = wikinetworkdata.edge_ids
    reverse_edge_id_list = wikinetworkdata.reverse_edge_ids

    n_train = int((1- (n_val + n_test)) * len(edge_id_list))
    n_val =  int(n_val * len(edge_id_list))

    random_i = torch.randperm(len(edge_id_list))

    train_edges = edge_id_list[random_i[:n_train]]
    train_edges_r = reverse_edge_id_list[random_i[:n_train]]

    val_edges = edge_id_list[random_i[n_train: (n_train + n_val)]]
    val_edges_r = reverse_edge_id_list[random_i[n_train: (n_train + n_val)]]

    test_edges = edge_id_list[random_i[(n_train + n_val):]]
    test_edges_r = reverse_edge_id_list[random_i[(n_train + n_val):]]

    mask = torch.zeros(len(torch.cat((edge_id_list, reverse_edge_id_list),0)), dtype=torch.bool)
    mask[torch.cat((train_edges,train_edges_r),0)] = True
    g.edata["train_mask"] = mask

    mask = torch.zeros(len(torch.cat((edge_id_list, reverse_edge_id_list),0)), dtype=torch.bool)
    mask[torch.cat((val_edges,val_edges_r),0)] = True
    g.edata["val_mask"] = mask

    mask = torch.zeros(len(torch.cat((edge_id_list, reverse_edge_id_list),0)), dtype=torch.bool)
    mask[torch.cat((test_edges,test_edges_r),0)] = True
    g.edata["test_mask"] = mask

    return train_edges, val_edges, test_edges



def randomise_node_features(g = None, random_node_frac=0.5, random_node_type="random", return_n = None):


    if return_n != None: ## simply return 2 random nodes from whole graph
        #random_nodes = torch.randperm(g.number_of_nodes())[:return_n]
        #return g.ndata["node_features"][random_nodes]
        #return torch.rand(return_n, 1, 500).float()
        return torch.ones(return_n, 1, 500).float()

    ## choose nodes
    num_random_nodes = int(random_node_frac * g.number_of_nodes())
    random_nodes = torch.randperm(g.number_of_nodes())[:num_random_nodes]

    ## construct random node features
    if random_node_type == "mean":
        random_node_features = torch.mean(g.ndata["node_features"], dim=0).view(1, -1)

    if random_node_type == "random":
        torch.manual_seed(1)
        #random_node_features = torch.rand(1,500).double()
        random_node_features = torch.ones(1, 500).float()

    ## assign random node features
    g.ndata["node_features"][random_nodes] = random_node_features



def randomise_edge_features(g = None, random_edge_frac=0.5, random_edge_type="random", return_n = None):

    if return_n != None: ## simply return 2 random nodes from whole graph
        #random_edges = torch.randperm(g.number_of_edges())[:return_n]
        #return g.edata["edge_features"][random_edges]
        #return torch.rand(return_n, 1, 500).float()
        return torch.ones(return_n,1,500).float()

    ## choose edges
    num_random_edges = int(random_edge_frac * g.number_of_edges())
    random_edges = torch.randperm(g.number_of_edges())[:num_random_edges]

    ## construct random edge features
    if random_edge_type == "mean":
        random_edge_features = torch.mean(g.edata["edge_features"], dim=0).view(1, -1)

    if random_edge_type == "random":
        torch.manual_seed(1)
        #random_edge_features = torch.rand(1,500).float()
        random_edge_features = torch.ones(1, 500).float()

    ## assign random edge features
    g.edata["edge_features"][random_edges] = random_edge_features




def randomise_labels(g, random_label_frac=0.5):

    ## choose edges
    num_random_labels = int(random_label_frac * g.number_of_edges())
    random_edges = torch.randperm(g.number_of_edges())[:num_random_labels]

    ## construct random edge features
    random_label_mask = torch.zeros((g.number_of_edges()))
    random_label_mask[random_edges] = True

    ## assign random edge features
    g.edata["label_mask"] = random_label_mask

