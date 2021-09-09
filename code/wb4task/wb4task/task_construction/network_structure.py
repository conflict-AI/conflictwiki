from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from collections import defaultdict


def create_edge_list(ally_enemy_pairs):
    edge_list_dict = defaultdict(list)

    for (entity1_id, entity2_id), c_id, label in ally_enemy_pairs:

        ## key – to ensure edges are included only once
        if entity1_id < entity2_id:
            dict_key = str(entity1_id) + "-" + str(entity2_id)
        else:
            dict_key = str(entity2_id) + "-" + str(entity1_id)

        ## value – sign
        if label == "enemies":
            dict_val = (-1, c_id)
        elif label == "allies":
            dict_val = (+1, c_id)

        edge_list_dict[dict_key].append(dict_val)

    return edge_list_dict



def aggregate_edge_list(edge_list_dict, conflict_id_name):

        aggr_edge_list = list()

        for i, (edge_id, edge_val) in enumerate(edge_list_dict.items()):

            nodes = edge_id.split("-")
            relation = 0
            c_id_list = list()

            for (rel, c_id) in edge_val:
                relation += rel
                c_id_list.append(c_id)

            if relation <= 0:
                relation_discrete = 0
                relation_word = "enemies"
            elif relation > 0:
                relation_discrete = 1
                relation_word = "allies"

            c_name_list = [conflict_id_name[c_id] for c_id in c_id_list]
            edge_triple = [int(nodes[0]), int(nodes[1]),{"edge_id": i, "label": relation_word, "label_discrete": relation_discrete, "label_continuous": relation, "n_conflicts": len(c_id_list), "conflict_ids": c_id_list, "conflict_names": c_name_list}]  # (node_from, node_to, dict(label, features))
            aggr_edge_list.append(edge_triple)

        return aggr_edge_list



def create_node_list(aggr_edge_list, entity_id_name):

    entity_list_raw = list(set([edge_item[0] for edge_item in aggr_edge_list] + [edge_item[1] for edge_item in aggr_edge_list]))
    node_list = list()

    for entity_id in entity_list_raw:
        node = (int(entity_id), {"name": entity_id_name[entity_id]})
        node_list.append(node)

    return node_list




if __name__ == "__main__":

    config = configs.ConfigBase()
    ally_enemy_pairs = load_file(config.get_path("task") / "ally_enemy_pairs", ftype="pkl")
    conflict_id_name = load_file(config.get_path("conflict_retrieval") / "conflict_id_name", ftype="pkl")
    entity_id_name = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")

    edge_list_dict = create_edge_list(ally_enemy_pairs)
    aggr_edge_list = aggregate_edge_list(edge_list_dict, conflict_id_name)
    node_list = create_node_list(aggr_edge_list, entity_id_name) ## build node list based on edges to ensure coverage

    store_file(config.get_path("task") / "network_structure" / "aggr_edge_list", aggr_edge_list, "pkl", "csv")
    store_file(config.get_path("task") / "network_structure" / "node_list", node_list, "pkl", "csv")