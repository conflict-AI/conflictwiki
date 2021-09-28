from wb0configs import configs
from wb0configs.helpers import load_file, store_file
from wb3embed.longformer.longformer_tokenizer_model import WikiLongformer

from collections import defaultdict
from tqdm import tqdm
import torch


def longformer_entity_embed(e_id_section, wikilongformer):

    e_id_embed = defaultdict(torch.Tensor) ## mean all sections per entity

    for e_id in tqdm(e_id_section.keys()):
        sections = e_id_section[e_id]

        for s_i, s_title in enumerate(sections.keys()):

            s_text = sections[s_title] ## section text
            id_embed = wikilongformer.text2embed(s_text)

            embed = id_embed[0].unsqueeze(0) #[[]] ## get  <CLS>
            e_id_embed[e_id] = torch.cat((e_id_embed[e_id], embed), 0)

        e_id_embed[e_id] = torch.mean(e_id_embed[e_id], 0) ## mean section embeddings

    return e_id_embed



if __name__ == "__main__":

    config = configs.ConfigBase()
    e_id_section = load_file(config.get_path("entity_dataprep") / "id_section", ftype="pkl")

    wikilongformer = WikiLongformer()
    e_id_embed = longformer_entity_embed(e_id_section, wikilongformer)

    store_file(config.get_path("entity_embed") / "e_id_embed", e_id_embed, "pkl")

