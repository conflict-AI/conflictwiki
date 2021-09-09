from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb4task.task_construction.ally_enemy_pairs import construct_ally_enemy_pairs

import torch
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import itertools


## mean across all sections

def preconstruct_entity_sec_task(gt, e_id_sec_embed):

    print("preconstruct_entity_sec_task")

    sec_sec_emb = load_file(config.get_path("task") / "entity_sec" / "pre" / "sec_sec_emb_tfidf", ftype="pkl")
    sec_sec_counts = load_file(config.get_path("task") / "entity_sec" / "pre" / "sec_sec_counts_tfidf", ftype="pkl")

    if sec_sec_emb == None:

        X_E = defaultdict(list)  ## enemy
        X_F = defaultdict(list)  ## friend
        sec_sec_E = list()
        sec_sec_F = list()

        for ((ent1, ent2), c_id, y) in tqdm(list(gt)):

            ent_1_sec_t_emb = list(e_id_sec_embed[ent1].items())
            ent_2_sec_t_emb = list(e_id_sec_embed[ent2].items())

            ## iterate over all section tuples between all entities
            for (ent_1_t, ent_1_emb), (ent_2_t, ent_2_emb) in list(itertools.product(ent_1_sec_t_emb, ent_2_sec_t_emb)):

                ## hash string for unique ordering
                ent_1_hash = hash(ent_1_t)
                ent_2_hash = hash(ent_2_t)
                if ent_1_hash > ent_2_hash:
                    dict_key = ent_1_t + " - " + ent_2_t
                    dict_val = (ent_1_emb, ent_2_emb)
                else:
                    dict_key = ent_2_t + " - " + ent_1_t
                    dict_val = (ent_2_emb, ent_1_emb)

                ## friend or enemy
                if y == "enemies":  ## enemy
                    X_E[dict_key].append(dict_val)
                    sec_sec_E.append(dict_key)
                else:  ## friend
                    X_F[dict_key].append(dict_val)
                    sec_sec_F.append(dict_key)

        sec_sec_emb = (X_E, X_F)
        sec_sec_counts = (sec_sec_E, sec_sec_F)

        store_file(config.get_path("task") / "entity_sec" / "pre" / "sec_sec_emb_tfidf", sec_sec_emb, "pkl")
        store_file(config.get_path("task") / "entity_sec" / "pre" / "sec_sec_counts_tfidf", sec_sec_counts, "pkl")

    return sec_sec_emb, sec_sec_counts


def get_most_frequent(X_E, X_F, sec_sec_E, sec_sec_F, k=100):

    print("get_most_frequent")

    sec_sec_E = [sec_sec_t for sec_sec_t, count in Counter(sec_sec_E).most_common(k)]
    X_E = {k: v for k, v in X_E.items() if k in sec_sec_E}

    sec_sec_F = [sec_sec_t for sec_sec_t, count in Counter(sec_sec_F).most_common(k)]
    X_F = {k: v for k, v in X_F.items() if k in sec_sec_F}

    return X_E, X_F


def construct_entity_sec_task(ally_enemy_pairs, e_id_sec_embed, k = 100):

    print("construct_entity_sec_task")

    data = load_file(config.get_path("task") / "entity_sec"  / "data", ftype="pkl")
    label = load_file(config.get_path("task") / "entity_sec"  / "label", ftype="pkl")

    test = False
    if test:#data != None:

        (X_1, X_2, Y) = data
        L = label

    else:

        (X_E, X_F), (sec_sec_E, sec_sec_F) = preconstruct_entity_sec_task(ally_enemy_pairs, e_id_sec_embed)
        X_E, X_F = get_most_frequent(X_E, X_F, sec_sec_E, sec_sec_F, k)

        emb_dim = 500#list(X_E.values())[0][0][0].shape[0]  ## 300 for w2v, 768 for Longformer

        ## create tensors
        n_e_sec = len(X_E.keys())
        print("n_e_sec:", n_e_sec)
        X_1_E = torch.FloatTensor(n_e_sec, emb_dim)
        X_2_E = torch.FloatTensor(n_e_sec, emb_dim)
        Y_E = torch.FloatTensor(n_e_sec * [0.0]).unsqueeze(1)
        L_E = list()

        for i, (sec_sec_key, emb_val) in tqdm(enumerate(X_E.items())):
            X_1_E[i, :] = torch.mean(torch.stack([x_emb[0] for x_emb in emb_val]), dim=0)
            X_2_E[i, :] = torch.mean(torch.stack([x_emb[1] for x_emb in emb_val]), dim=0)
            L_E.append(sec_sec_key)

        n_f_sec = len(X_F.keys())
        print("n_f_sec:", n_f_sec)
        X_1_F = torch.FloatTensor(n_f_sec, emb_dim)
        X_2_F = torch.FloatTensor(n_f_sec, emb_dim)
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

        store_file(config.get_path("task") / "entity_sec"  / "data", (X_1, X_2, Y), "pkl")
        store_file(config.get_path("task") / "entity_sec"  / "label", L, "pkl")

    return (X_1, X_2, Y), L



if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    #e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed", ftype="pkl")
    e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed_tfidf", ftype="pkl")

    ally_enemy_pairs = construct_ally_enemy_pairs(config, c_e_id)
    data, labels = construct_entity_sec_task(ally_enemy_pairs, e_id_sec_embed, k=1000)



