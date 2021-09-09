from wb0configs import configs
from wb0configs.helpers import load_file, store_file
from wb3embed.longformer.longformer_tokenizer_model import WikiLongformer

import torch
from collections import defaultdict
from tqdm import tqdm


def longformer_conflict_embed(c_id_section, wikilongformer):
    ## mean all section and entity embeddings on the article level

    c_id_ent_id_embed = defaultdict(dict)
    #c_id_embed = defaultdict(torch.Tensor)

    for c_id in tqdm(c_id_section.keys()):
        sections = c_id_section[c_id]
        ent_id_embed = defaultdict(torch.Tensor)

        for s_i, s_title in enumerate(sections.keys()):

            s_text = sections[s_title]   ## section text
            id_embed = wikilongformer.text2embed(s_text)

            for ent_id, embed in id_embed.items():
                embed = embed.unsqueeze(0) #[[]]
                c_id_ent_id_embed[int(c_id)][int(ent_id)] = torch.cat((ent_id_embed[int(ent_id)], embed),0)  ## concat all representations

        ## take mean of embeddings across sections
        for ent_id, embed in c_id_ent_id_embed[int(c_id)].items():
            c_id_ent_id_embed[int(c_id)][int(ent_id)] = torch.mean(c_id_ent_id_embed[int(c_id)][int(ent_id)],0)

    return c_id_ent_id_embed



if __name__ == "__main__":

    config = configs.ConfigBase()
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    c_id_section = load_file(config.get_path("conflict_dataprep") / "id_section", ftype="pkl")

    wikilongformer = WikiLongformer(entity_dict)
    c_id_ent_id_embed = longformer_conflict_embed(c_id_section, wikilongformer)
    store_file(config.get_path("conflict_embed") / "c_id_ent_id_embed", c_id_ent_id_embed, "pkl")

