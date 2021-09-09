from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml

import itertools
import torch
from tqdm import tqdm
from itertools import chain

def construct_ground_truth(config, c_e_id, c_id = None, e_id = None):

    print("construct_ground_truth")
    gt = load_file(config.get_path("task") / "ground_truth", ftype="pkl")

    if gt == None or c_id != None or e_id != None:
        enemies = list()
        friends = list()


        if c_id != None:
            c_e_id = {k: v for (k,v) in c_e_id.items() if k in c_id}

        for c_id in list(c_e_id.keys()):

            combatants = c_e_id[c_id]
            n_bell = len(combatants)
            all_entities = list(chain(*combatants))

            ## enemies
            for combatant_pair in itertools.combinations(combatants, r=2):
                for enemy_pair in itertools.product(*combatant_pair):
                    all_other_ent = list(set(all_entities) - set(enemy_pair))
                    c_enemy = (c_id, enemy_pair, all_other_ent, n_bell, 0)
                    enemies.append(c_enemy)

                    if e_id == None:
                        friends.append(c_enemy)
                    elif e_id[0] in enemy_pair:
                        friends.append(c_enemy)

            for alliance in combatants:
                ## friends
                for friend_pair in itertools.combinations(alliance, r=2):
                    all_other_ent = list(set(all_entities) - set(friend_pair))
                    c_friend = (c_id, friend_pair, all_other_ent, n_bell, 1)

                    if e_id == None:
                        friends.append(c_friend)
                    elif e_id[0] in friend_pair:
                        friends.append(c_friend)

        gt = enemies + friends
        store_file(config.get_path("task") / "ground_truth", gt, "pkl", "csv")

    return gt




if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    gt = ally_enemy_pairs(config, c_e_id, c_id = None, e_id = None)
    print(gt)