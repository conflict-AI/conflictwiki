from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from collections import defaultdict
import random
import torch
from tqdm import tqdm


def pool_sections_embed(e_id_sec_embed):
    print("pool_sections_embed")

    e_id_embed = load_file(config.get_path("task") / "entity_pooled" / "e_id_embed_tfidf", ftype="pkl")
    #e_id_embed = load_file(config.get_path("task") / "entity_pooled" / "e_id_embed_tfidf", ftype="pkl")

    if e_id_embed == None:
        e_id_embed = dict()

        for e, sec in e_id_sec_embed.items():
            e_id_embed[e] = torch.mean(torch.stack(list(sec.values())), dim=0)

        store_file(config.get_path("task") / "entity_pooled" / "e_id_embed_tfidf", e_id_embed, "pkl", "csv")
    return e_id_embed



def create_node_features(node_list, e_id_embed):
    print("create_node_features")

    #node_features = torch.Tensor(len(node_list),768)
    node_features = dict()

    for i, (e_id, e_features) in enumerate(node_list):
        if e_id in e_id_embed.keys():
            e_emb = e_id_embed[e_id]
            #node_features[i] = e_emb
            node_features[e_id] = e_emb
        else: ## for some entities there exist no embeddings
            e_id= random.choice(node_list)[0]
            e_emb = e_id_embed[e_id]
            #node_features[i] = e_emb
            node_features[e_id] = e_emb

    return node_features



def create_edge_features(aggr_edge_list, c_id_ent_id_embed):
    print("create_edge_features")

    #edge_features = torch.Tensor(len(aggr_edge_list), 768)
    edge_features = dict()

    for i, (e_id_1, e_id_2, c_features) in tqdm(enumerate(aggr_edge_list)):
        c_ids = c_features["conflict_ids"]
        for c_id in c_ids:
            #edge_features[i] = c_id_ent_id_embed[c_id][0]
            edge_features[(e_id_1, e_id_2)] = c_id_ent_id_embed[c_id]

    return edge_features


if __name__ == "__main__":

    config = configs.ConfigBase()

    node_list = load_file(config.get_path("task") / "network_structure" / "node_list", ftype = "pkl")
    e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed_tfidf", ftype="pkl")
    e_id_embed = pool_sections_embed(e_id_sec_embed)

    node_features = create_node_features(node_list, e_id_embed)

    aggr_edge_list = load_file(config.get_path("task") / "network_structure" / "aggr_edge_list", ftype = "pkl")
    c_id_ent_id_embed = load_file(config.get_path("conflict_embed") / "c_id_ent_id_embed_tfidf", ftype="pkl")
    edge_features = create_edge_features(aggr_edge_list, c_id_ent_id_embed)

    store_file(config.get_path("task") / "network_features" / "node_features", node_features, "pkl", "csv")
    store_file(config.get_path("task") / "network_features" / "edge_features", edge_features, "pkl", "csv")

