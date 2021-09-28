from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import networkx as nx



def add_nodes(nx_graph, node_list, node_features):

    enriched_node_list = list()

    for node in node_list:
        node_id = node[0]
        node[1]["entity_id"] = node_id
        node[1]["node_features"] = node_features[node_id]
        enriched_node_list.append((node_id, node[1]))

    nx_graph.add_nodes_from(enriched_node_list)
    return nx_graph



def add_edges(nx_graph, aggr_edge_list, edge_features):

    enriched_aggr_edge_list = list()

    for edge in aggr_edge_list:
        edge_u = edge[0]
        edge_v = edge[1]
        edge[2]["edge_id"] = (edge_u, edge_v)
        edge[2]["edge_features"] = edge_features[(edge_u, edge_v)]
        enriched_aggr_edge_list.append((edge_u, edge_v, edge[2]))

    nx_graph.add_edges_from(enriched_aggr_edge_list)
    return nx_graph


def construct_networkx(config, undirected = True, remove_n_degr_below = 1, remove_clust_coef_below = 0.0):

    ## structure
    node_list = load_file(config.get_path("task") / "network_structure" / "node_list", ftype="pkl")
    aggr_edge_list = load_file(config.get_path("task") / "network_structure" / "aggr_edge_list", ftype="pkl")

    ## features
    node_features = load_file(config.get_path("task") / "network_features" / "node_features", ftype="pkl")
    edge_features = load_file(config.get_path("task") / "network_features" / "edge_features", ftype="pkl")

    if undirected:
        nx_graph = nx.Graph()
    else:
        nx_graph = nx.DiGraph()

    nx_graph = add_nodes(nx_graph, node_list, node_features)
    nx_graph = add_edges(nx_graph, aggr_edge_list, edge_features)

    #nx.set_node_attributes(nx_graph, node_features, "node_features")
    #nx.set_edge_attributes(nx_graph, edge_features, "edge_features")

    nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    if undirected:
        nx_graph = nx_graph.to_undirected()

    remove = [node for node, degree in dict(nx_graph.degree()).items() if degree < remove_n_degr_below]
    remove += [node for node, clust_coef in nx.clustering(nx_graph).items() if clust_coef < remove_clust_coef_below]
    nx_graph.remove_nodes_from(remove)

    return nx_graph




if __name__ == "__main__":

    config = configs.ConfigBase()
    nx_graph = construct_networkx(config, undirected= True, remove_n_degr_below = 1, remove_clust_coef_below = 0.25)
    print("nodes:", len(nx_graph.nodes()))
    print("edges:", len(nx_graph.edges()))
