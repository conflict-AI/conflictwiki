import spacy
from torchtext.vocab import Vectors
import torch
import nltk
import numpy as np
from tqdm import tqdm
import spacy

NLP = spacy.load("en_core_web_sm")


#nltk.data.path.append('/Users/niklasstoehr/Libraries/nltk_data')
spacy_en = spacy.load('en')


def init_word2vec(config, type):
    word2vec = Vectors(config.get_path("word2vec") / str(type))
    #word2vec = torchtext.vocab.GloVe(name='840B', dim=300)
    return word2vec


def embedder(token_list, word2vec):
    emb_tensor = torch.FloatTensor(len(token_list), 300)

    for i, token in enumerate(token_list):
        emb = word2vec[str(token)]
        emb_tensor[i] = emb
    return emb_tensor


def aggregate_embeddings(emb_tensor, token_list, aggr="mean"):
    emb_mean = torch.mean(emb_tensor, axis=0)
    return emb_mean


def tokenise_ner_removal(text):

    tokens = list()
    doc = NLP(text)

    ## keep NORP: German, Republican, Christianity, but not Taliban, Al-QAEDA
    entity_blacklist = ["ORG", "GPE", "PERSON", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
                        "DATE", "MONEY"]

    for tok in doc:
        if tok.ent_type_ not in entity_blacklist:  ## remove named entities
            if tok.is_stop != True:  ## remove stop words
                if tok.is_punct != True:  ## remove punctuation
                    tok_lemma = tok.lemma_  ## lemmatise
                    tokens.append(tok_lemma)

    return tokens




def get_corpus_frac_dist(entities):
    print("get_corpus_frac_dist")
    global_token_list = list()

    for entity_id, sec_dict in tqdm(entities.items()):
        for sec_title, sec_text in sec_dict.items():

            token_list = tokenise_ner_removal(sec_text) ## use text
            if len(token_list) == 0:
                token_list = tokenise_ner_removal(sec_title) ## use title

            global_token_list += token_list

    ## compute frequency distribution
    total_token_count = len(global_token_list)
    freq_dict = nltk.FreqDist(global_token_list)

    frac_dict = {}
    for token, count in freq_dict.items():
        frac_dict[token] = (count, count / total_token_count)

    return frac_dict



def corpus_quantile(frac_dict, lower_quant=0.1, upper_quant=0.75):
    frac_dict_filtered = dict()

    frac_list = np.asarray(list(frac_dict.values()))[:, 1]
    upper_quant = np.quantile(np.asarray(frac_list), upper_quant)
    lower_quant = np.quantile(np.asarray(frac_list), lower_quant)

    for term, (count, prob) in frac_dict.items():

        if prob > lower_quant and prob < upper_quant:
            frac_dict_filtered[term] = (count, prob)

    return frac_dict_filtered




def filter_tokenizer(sec_title, sec_text, frac_dict):

    filtered_token_list = list()
    token_list = tokenise_ner_removal(sec_text)

    ## section text
    if len(token_list) > 0:
        for token in token_list:
            if token in frac_dict.keys():
                filtered_token_list.append(token)

    ## section title
    if len(filtered_token_list) == 0:
        token_list = tokenise_ner_removal(sec_title)
        if len(token_list) > 0:
            for token in token_list:
                if token in frac_dict.keys():
                    filtered_token_list.append(token)

    return filtered_token_list