from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb4task.task_construction.ally_enemy_pairs import construct_ally_enemy_pairs

import itertools
import torch
from tqdm import tqdm
from itertools import chain
from collections import defaultdict, Counter


## all sections per enities per conflict

def preconstruct_entity_sec_task(gt, e_id_sec_embed, entity_dict):
    X_E = defaultdict(list)  ## enemy
    X_F = defaultdict(list)  ## friend
    sec_sec_E = list()
    sec_sec_F = list()
    ent_ent_E = list()
    ent_ent_F = list()


    for ((ent1, ent2), c_id, y) in tqdm(gt):

        ent1_name = entity_dict[ent1]
        ent2_name = entity_dict[ent2]

        ent_1_sec_t_emb = list(e_id_sec_embed[ent1].items())
        ent_2_sec_t_emb = list(e_id_sec_embed[ent2].items())

        ## iterate over all section tuples between entities
        for (ent_1_t, ent_1_emb), (ent_2_t, ent_2_emb) in list(itertools.product(ent_1_sec_t_emb, ent_2_sec_t_emb)):

            ## hash string for unique ordering
            ent_1_hash = hash(ent_1_t)
            ent_2_hash = hash(ent_2_t)
            if ent_1_hash > ent_2_hash:
                dict_key = ent_1_t + " - " + ent_2_t
                ent_name = ent1_name + " - " + ent2_name
                dict_val = (ent_1_emb, ent_2_emb)
            else:
                dict_key = ent_2_t + " - " + ent_1_t
                ent_name = ent2_name + " - " + ent1_name
                dict_val = (ent_2_emb, ent_1_emb)

            ## friend or enemy
            if y == 0.0:  ## enemy
                X_E[dict_key].append(dict_val)
                sec_sec_E.append(dict_key)
                ent_ent_E.append(ent_name)
            else:  ## friend
                X_F[dict_key].append(dict_val)
                sec_sec_F.append(dict_key)
                ent_ent_F.append(ent_name)

    return X_E, X_F, sec_sec_E, ent_ent_E, sec_sec_F, ent_ent_F


def get_most_frequent(X_E, X_F, sec_sec_E, sec_sec_F, k=100):
    sec_sec_E = [sec_sec_t for sec_sec_t, count in Counter(sec_sec_E).most_common(k)]
    X_E = {k: v for k, v in X_E.items() if k in sec_sec_E}

    sec_sec_F = [sec_sec_t for sec_sec_t, count in Counter(sec_sec_F).most_common(k)]
    X_F = {k: v for k, v in X_F.items() if k in sec_sec_F}

    return X_E, X_F


def construct_entity_sec_task(X_E, X_F):
    ## create tensors
    n_e_sec = len(X_E.keys())
    print("n_e_sec:", n_e_sec)
    X_1_E = torch.FloatTensor(n_e_sec, 768)
    X_2_E = torch.FloatTensor(n_e_sec, 768)
    Y_E = torch.FloatTensor(n_e_sec * [0.0]).unsqueeze(1)
    L_E = list()

    for i, (sec_sec_key, emb_val) in tqdm(enumerate(X_E.items())):
        X_1_E[i, :] = torch.mean(torch.stack([x_emb[0] for x_emb in emb_val]), dim=0)
        X_2_E[i, :] = torch.mean(torch.stack([x_emb[1] for x_emb in emb_val]), dim=0)
        L_E.append(sec_sec_key)

    n_f_sec = len(X_F.keys())
    print("n_f_sec:", n_f_sec)
    X_1_F = torch.FloatTensor(n_f_sec, 768)
    X_2_F = torch.FloatTensor(n_f_sec, 768)
    Y_F = torch.FloatTensor(n_f_sec * [1.0]).unsqueeze(1)
    L_F = list()

    for i, (sec_sec_key, emb_val) in tqdm(enumerate(X_F.items())):
        X_1_F[i, :] = torch.mean(torch.stack([x_emb[0] for x_emb in emb_val]), dim=0)
        X_2_F[i, :] = torch.mean(torch.stack([x_emb[1] for x_emb in emb_val]), dim=0)
        L_F.append(sec_sec_key)

    ## combine data
    X_1 = torch.cat((X_1_E, X_1_F), dim=0)
    X_2 = torch.cat((X_2_E, X_2_F), dim=0)
    Y = torch.cat((Y_E, Y_F), dim=0)
    L = L_E + L_F

    return (X_1, X_2, Y), L


if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    #e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed", ftype="pkl")
    e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed_w2v", ftype="pkl")
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    ally_enemy_pairs = construct_ally_enemy_pairs(config, c_e_id, c_id = [15802807])

    X_E, X_F, sec_sec_E, ent_ent_E, sec_sec_F, ent_ent_F = preconstruct_entity_sec_task(ally_enemy_pairs, e_id_sec_embed, entity_dict)
    X_E, X_F = get_most_frequent(X_E, X_F, sec_sec_E, sec_sec_F, k=100)
    data, labels = construct_entity_sec_task(X_E, X_F)