from wb0configs import configs
from wb0configs.helpers import load_file, store_file
from wb3embed.word2vec.word2vec_embedder import embedder, aggregate_embeddings, init_word2vec, get_corpus_frac_dist, corpus_quantile, filter_tokenizer

from collections import defaultdict
from tqdm import tqdm


def word2vec_entity_embed(e_id_section):

    frac_dict = get_corpus_frac_dist(e_id_section)
    frac_dict = corpus_quantile(frac_dict, lower_quant=0.1, upper_quant=0.75)

    e_id_sec_embed = defaultdict(dict)
    word2vec = init_word2vec(config, type = "glove.840B.300d.txt")

    for entity_id, sec_dict in tqdm(e_id_section.items()):
        for sec_title, sec_text in sec_dict.items():

            token_list = filter_tokenizer(sec_title, sec_text, frac_dict)

            if len(token_list) > 0:
                emb_tensor = embedder(token_list, word2vec)
                emb_mean = aggregate_embeddings(emb_tensor, token_list, aggr="mean")
                e_id_sec_embed[entity_id][sec_title] = emb_mean

    return e_id_sec_embed




def longformer_entity_embed(e_id_section, wikilongformer):

    e_id_sec_embed = defaultdict(dict) ## mean section embedding per entity
    #e_id_embed = defaultdict(torch.Tensor) ## mean all sections per entity

    for e_id in tqdm(e_id_section.keys()):
        sections = e_id_section[e_id]

        for s_i, s_title in enumerate(sections.keys()):

            s_text = sections[s_title] ## section text
            id_embed = wikilongformer.text2embed(s_text)

            #embed = id_embed[0].unsqueeze(0) #[[]] ## get  <CLS>
            e_id_sec_embed[e_id][s_title] = id_embed[0]
            #e_id_embed[e_id] = torch.cat((e_id_embed[e_id], id_embed[0].unsqueeze(0)), 0)

        #e_id_embed[e_id] = torch.mean(e_id_embed[e_id], 0) ## mean section embeddings

    return e_id_sec_embed




if __name__ == "__main__":

    config = configs.ConfigBase()
    e_id_section = load_file(config.get_path("entity_dataprep") / "id_section", ftype="pkl")

    #wikilongformer = WikiLongformer()
    #e_id_sec_embed = longformer_entity_embed(e_id_section, wikilongformer)

    e_id_sec_embed = word2vec_entity_embed(e_id_section)

    #store_file(config.get_path("entity_embed") / "e_id_embed", e_id_embed, "pkl")
    store_file(config.get_path("entity_embed") / "e_id_sec_embed_w2v", e_id_sec_embed, "pkl")

