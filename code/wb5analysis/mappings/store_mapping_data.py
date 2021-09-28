from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import pandas as pd


def store_ally_enemy_pairs(config, ally_enemy_pairs):

    df = pd.DataFrame(ally_enemy_pairs, columns = ["entity_pair", "conflict_id", "entity_relationship"])
    df.to_csv(config.get_path("data_publishing") / "mappings" / "ally_enemy_pairs" / "ally_enemy_pairs.csv", index=True)
    store_file(config.get_path("data_publishing") / "mappings" / "ally_enemy_pairs" / "ally_enemy_pairs", ally_enemy_pairs, "pkl")
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "mappings" / "ally_enemy_pairs" / "ally_enemy_pairs.json", index=True,  force_ascii=False, lines = True, orient = "records")
    print(df)


def store_c_e_id(config, c_e_id):

    df = pd.DataFrame(list(zip(list(c_e_id.keys()), list(c_e_id.values()))), columns = ["conflict_id", "entity_ids"])
    df = df.set_index("conflict_id")
    print(df)

    store_file(config.get_path("data_publishing") / "mappings" / "conflict_entity_id" / "conflict_entity_id", c_e_id, "pkl")
    df.to_csv(config.get_path("data_publishing") / "mappings" / "conflict_entity_id" / "conflict_entity_id.csv", index=True)
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "mappings" / "conflict_entity_id" / "conflict_entity_id.json", index=True,  force_ascii=False, lines = True, orient = "records")


def store_network_structure(config, aggr_edge_list, node_list):

    df = pd.DataFrame(aggr_edge_list, columns = ["entity_id_1", "entity_id_2", "conflict_attributes"])
    print(df)

    df.to_csv(config.get_path("data_publishing") / "mappings" / "network" / "aggr_edge_list.csv", index=False)
    store_file(config.get_path("data_publishing") / "mappings" / "network" / "aggr_edge_list", aggr_edge_list, "pkl")
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "mappings" / "network" / "aggr_edge_list.json", index=True,  force_ascii=False, lines = True, orient = "records")

    df = pd.DataFrame(node_list, columns = ["entity_id", "entity_attributes"])
    print(df)

    df.to_csv(config.get_path("data_publishing") / "mappings" / "network" / "node_list.csv", index=False)
    store_file(config.get_path("data_publishing") / "mappings" / "network" / "node_list", node_list, "pkl")
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "mappings" / "network" / "node_list.json", index=True,  force_ascii=False, lines = True, orient = "records")



if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    ally_enemy_pairs = load_file(config.get_path("task") / "ally_enemy_pairs", ftype="pkl")

    aggr_edge_list = load_file(config.get_path("task") / "network_structure" / "aggr_edge_list", ftype="pkl")
    node_list = load_file(config.get_path("task") / "network_structure" / "node_list", ftype="pkl")

    ## call store functions
    store_ally_enemy_pairs(config, ally_enemy_pairs)
    store_c_e_id(config, c_e_id)
    store_network_structure(config, aggr_edge_list, node_list)
