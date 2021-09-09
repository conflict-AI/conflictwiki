from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import itertools


def construct_ally_enemy_pairs(config, c_e_id, c_id = None, e_id = None):

    print("ally_enemy_pairs")
    ally_enemy_pairs = load_file(config.get_path("task") / "ally_enemy_pairs", ftype="pkl")

    if ally_enemy_pairs == None or c_id != None or e_id != None:

        ally_enemy_pairs = list()

        if c_id != None:
            c_e_id = {k: v for (k,v) in c_e_id.items() if k in c_id}


        for c_id in list(c_e_id.keys()):

            combatants = c_e_id[c_id]
            #n_bell = len(combatants)
            #all_entities = list(itertools.chain(*combatants))

            ## enemies
            for combatant_pair in itertools.combinations(combatants, r=2):
                for enemy_pair in itertools.product(*combatant_pair):
                    ally_enemy_pairs.append((enemy_pair, c_id, "enemies"))

            for alliance in combatants:

                ## allies
                for friend_pair in itertools.combinations(alliance, r=2):
                    ally_enemy_pairs.append((friend_pair, c_id, "allies"))

        store_file(config.get_path("task") / "ally_enemy_pairs", ally_enemy_pairs, "pkl", "csv")

    return ally_enemy_pairs


if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    ally_enemy_pairs = construct_ally_enemy_pairs(config, c_e_id, c_id = None, e_id = None)
