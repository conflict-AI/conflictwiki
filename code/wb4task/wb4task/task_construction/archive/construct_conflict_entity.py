from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml
from wb4task.task_construction.ally_enemy_pairs import construct_ally_enemy_pairs

import torch
from tqdm import tqdm



def pool_sections_embed(e_id_sec_embed):

    print("pool_sections_embed")
    e_id_embed = load_file(config.get_path("task") / "entity_pooled" / "e_id_embed_w2v", ftype="pkl")

    if e_id_embed == None:
        e_id_embed = dict()

        for e, sec in e_id_sec_embed.items():
            e_id_embed[e] = torch.mean(torch.stack(list(sec.values())), dim=0)

        store_file(config.get_path("task") / "entity_pooled" / "e_id_embed_w2v", e_id_embed, "pkl", "csv")

    return e_id_embed



def construct_conflict_entity_task(gt, c_id_ent_id_embed, e_id_embed):

    print("construct_conflict_entity_task")

    data = load_file(config.get_path("task") / "conflict_entity_task"  / "data", ftype="pkl")
    label = load_file(config.get_path("task") / "conflict_entity_task"  / "label", ftype="pkl")

    if data != None:

        (X_1, X_2, C, E, Y) = data
        L = label

    else:

        L = list()  ## labels
        X_1 = list()  ## entity 1 in conflict A
        X_2 = list()  ## entity 2 in conflict A
        C = list()  ## conflict
        E = list()  ## all entities
        Y = list()  ## target

        for (c_id, (ent1, ent2), all_other_ent, n_bell, y) in tqdm(gt):

            ## label
            l = torch.LongTensor([[c_id, ent1, ent2, len(all_other_ent), n_bell, y]])
            L.append(l)

            ## entity 1 in conflict article
            if ent1 in c_id_ent_id_embed[c_id].keys():  ## check if entity in conflict article
                c_ent1_embed = c_id_ent_id_embed[c_id][ent1].unsqueeze(0)
            else:  ## some entities are not in conflict article
                c_ent1_embed = c_id_ent_id_embed[c_id][0].unsqueeze(0)  ## then take entire article representation

            ent1_embed = e_id_embed[ent1].unsqueeze(0)  ## entity 1 article
            ent1_embed_both = torch.cat((c_ent1_embed, ent1_embed), 0).unsqueeze(0)  ## shape 1,2,768

            X_1.append(ent1_embed_both)

            ## entity 2 in conflict article
            if ent2 in c_id_ent_id_embed[c_id].keys():  ## check if entity in conflict article
                c_ent2_embed = c_id_ent_id_embed[c_id][ent2].unsqueeze(0)
            else:  ## some entities are not in conflict article
                c_ent2_embed = c_id_ent_id_embed[c_id][0].unsqueeze(
                    0)  ## then take entire article representation

            ent2_embed = e_id_embed[ent2].unsqueeze(0)  ## entity 2 article
            ent2_embed_both = torch.cat((c_ent2_embed, ent2_embed), 0).unsqueeze(0)  ## shape 1,2,768

            X_2.append(ent2_embed_both)

            ## conflict article
            c = c_id_ent_id_embed[c_id][0].unsqueeze(0)
            C.append(c)

            ## all entities
            mean_ent_emb = torch.FloatTensor()
            if len(all_other_ent) > 0:
                for ent_id in all_other_ent:  ## all other entities
                    ent_emb = e_id_embed[ent_id].unsqueeze(0)
                    mean_ent_emb = torch.cat((mean_ent_emb, ent_emb), 0)
                mean_ent_emb = torch.mean(mean_ent_emb, dim=0).unsqueeze(0)
            else:  ## one-on-one conflict
                mean_ent_emb = torch.cat((ent1_embed, ent2_embed), 0)
                mean_ent_emb = torch.mean(mean_ent_emb, dim=0).unsqueeze(0)

            E.append(mean_ent_emb)

            ## target
            Y.append(torch.Tensor([y]))

        ## list to tensor
        X_1 = torch.stack(X_1)
        X_1 = X_1.squeeze(1)
        X_2 = torch.stack(X_2)
        X_2 = X_2.squeeze(1)
        C = torch.stack(C)
        C = C.squeeze(1)
        E = torch.stack(E)
        E = E.squeeze(1)
        Y = torch.stack(Y)
        #Y = Y.squeeze(1)
        L = torch.stack(L)
        L = L.squeeze(1)

        store_file(config.get_path("task") / "conflict_entity_task"  / "data", (X_1, X_2, C, E, Y), "pkl")
        store_file(config.get_path("task") / "conflict_entity_task"  / "label", L, "pkl")

    return (X_1, X_2, C, E, Y), L  ## entity in conflict, entity



if __name__ == "__main__":

    config = configs.ConfigBase()
    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    e_id_sec_embed = load_file(config.get_path("entity_embed") / "e_id_sec_embed_w2v", ftype="pkl")

    gt = construct_ally_enemy_pairs(config, c_e_id)
    e_id_embed = pool_sections_embed(e_id_sec_embed)

    c_id_ent_id_embed = load_file(config.get_path("conflict_embed") / "c_id_ent_id_embed", ftype="pkl")
    data, labels = construct_conflict_entity_task(gt, c_id_ent_id_embed, e_id_embed)


