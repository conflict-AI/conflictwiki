import numpy as np
from tqdm import tqdm
from itertools import islice
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import torch
import re

from wb0configs import configs
from wb0configs.helpers import load_file, store_file


NLP = spacy.load("en_core_web_sm")


def tokenise_ner_removal(text):

    tokens = list()
    doc = NLP(text)

    ## keep NORP: German, Republican, Christianity, but not Taliban, Al-QAEDA
    entity_blacklist = ["ORG", "GPE", "PERSON", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                        "DATE", "MONEY", "TIME", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"]

    allowed_norp = ["christian", "christians", "jews", "jewish", "muslim", "islam", "muslims", "hindi", "christianity"]

    for tok in doc:
        if tok.pos_ in ["NOUN", "ADJ", "PROPN"]:  ## keep only nouns, proper nouns and adjectives
            if tok.ent_type_ == "" or tok.text.lower() in allowed_norp:  ## remove named entities
            #if tok.ent_type_ not in entity_blacklist:  ## remove named entities
                if tok.is_stop != True:  ## remove stop words
                    if tok.is_punct != True:  ## remove punctuation
                        if tok.text.isalpha():  ## only text chars
                            if len(tok.text) >= 3:  ## only longer than 3 characters
                                tok_lemma = tok.lemma_  ## lemmatise
                                tokens.append(tok_lemma.lower())
    return tokens



def build_flat_token_list(e_id_section):

    print("build_flat_token_list")
    entsec_toks = list()

    for entity_id, sec_dict in tqdm(e_id_section.items()):
        for sec_title, sec_text in sec_dict.items():

            token_list = tokenise_ner_removal(sec_text) ## use text
            if len(token_list) == 0:
                token_list = tokenise_ner_removal(sec_title) ## use title

            entsec_toks.append(token_list)
    return entsec_toks



def initialise_tfidf_vectoriser(corpus):

    print("initialise_tfidf_vectoriser")
    def identity_override(text):
        return text

    tfidf_model = TfidfVectorizer(preprocessor = identity_override, tokenizer = identity_override, max_df = 0.4, min_df=0.01, max_features=500)
    vectors = tfidf_model.fit_transform(corpus)

    feature_names = tfidf_model.get_feature_names()
    features = dict(zip(range(0, len(feature_names)), feature_names))
    print("feature_names:", feature_names)
    return vectors, features



def set_vectors(e_id_section, entsec_vecs):

    print("set_vectors")

    e_id_vec = e_id_section.copy()
    i = 0

    for entity_id, sec_dict in tqdm(e_id_section.items()):
        for sec_title, sec_text in sec_dict.items():
            vec = torch.from_numpy(entsec_vecs[i].todense())
            e_id_vec[entity_id][sec_title] = vec.squeeze(0)
            i += 1
    return e_id_vec



def tfidf_embed(dict_dict_str):

    flat_toks = build_flat_token_list(dict_dict_str)
    flat_vecs, features = initialise_tfidf_vectoriser(flat_toks)
    dict_dict_vec = set_vectors(dict_dict_str, flat_vecs)

    return dict_dict_vec, flat_toks, features


def pool_section_emb(dict_dict_vec):
    print("pool_section_emb")

    dict_vec = dict()
    for art_id, sec_dict in tqdm(dict_dict_vec.items()):
        art_vec = torch.FloatTensor(len(sec_dict),list(sec_dict.values())[0].shape[0]) ## sections, n_features
        for i, (sec_title, sec_text) in enumerate(sec_dict.items()):
            art_vec[i] = dict_dict_vec[art_id][sec_title]
        dict_vec[art_id] = torch.mean(art_vec,0)
    return dict_vec



def entity_embedding(e_id_section):

    ent_id_vec, ent_toks, ent_features = tfidf_embed(e_id_section)
    store_file(config.get_path("entity_embed") / "e_id_sec_embed_tfidf", ent_id_vec, "pkl")

    ent_id_vec_pooled = pool_section_emb(ent_id_vec)
    store_file(config.get_path("task") / "entity_pooled" / "e_id_embed_tfidf", ent_id_vec_pooled, "pkl")

    #store_file(config.get_path("entity_embed") / "ent_toks", ent_toks, "pkl")
    #store_file(config.get_path("entity_embed") / "e_tfidf_features", ent_features, "pkl")


def conflict_embedding(c_id_section):
    c_id_vec, c_toks, c_features = tfidf_embed(c_id_section)
    c_id_vec = pool_section_emb(c_id_vec)
    store_file(config.get_path("conflict_embed") / "c_id_ent_id_embed_tfidf", c_id_vec, "pkl")
    store_file(config.get_path("conflict_embed") / "c_toks", c_toks, "pkl")
    store_file(config.get_path("conflict_embed") / "c_tfidf_features", c_features, "pkl")



if __name__ == "__main__":

    config = configs.ConfigBase()
    e_id_section = load_file(config.get_path("entity_dataprep") / "id_section", ftype="pkl")
    c_id_section = load_file(config.get_path("conflict_dataprep") / "id_section", ftype="pkl")

    #entity_embedding(dict(islice(e_id_section.items(), 2)))
    entity_embedding(e_id_section)
    #conflict_embedding(dict(islice(c_id_section.items(), 1)))
    #conflict_embedding(c_id_section)



