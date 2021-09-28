from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml

import spacy
import neuralcoref
import re
from tqdm import tqdm
from countryinfo import CountryInfo  #coding=utf-8
from pathlib import Path

nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)


def spacy_entity_coreference(text, entity_list):

    ## add entities to neuralcoref
    for entity in entity_list:
        nlp.get_pipe('neuralcoref').set_conv_dict({f'<{entity}/>': ['entity', 'country', 'organization']})

    ## peform entity coreference resolution
    nlp_doc = nlp(text)
    text = nlp_doc._.coref_resolved

    return text


def regex_entity_coreference(text, entity_list, redirect_entity):

    for entity in entity_list:
        text = re.sub(rf"([^<])({entity})([^/>])", r'\1<\2/>\3', text, flags=re.IGNORECASE)  ## consider all entities

        if entity in redirect_entity.values(): ## consider redirect entities in [[ ]]
            relevant_redirect_entity = list(set(re.findall(r"(?=(" + '|'.join(list(redirect_entity.keys())) + r"))", text)))

            if len(relevant_redirect_entity) > 0:
                for re_entity in relevant_redirect_entity:
                    text = re.sub(rf"([^<])({re_entity})([^/>])", rf"\1<{redirect_entity[re_entity]}/>\3", text, flags=re.IGNORECASE)  ## consider all entities
    return text



def country_demonym_resolution(text, entity_list, country_demonyms_dict):

    for entity in entity_list:
        text = re.sub(rf"([^<])({entity})([^/>])", r'\1<\2/>\3', text, flags= re.IGNORECASE | re.DOTALL)  ## consider all entities
        country_demonyms = country_demonyms_dict[entity]
        for country_demonym in country_demonyms: ## consider redirect entities
            if len(country_demonym) > 3: ## avoid accronyms such as RUS for Russia
                text = re.sub(rf"([^<])({country_demonym})([^/>])", rf"\1<{entity}/>\3", text, flags= re.IGNORECASE | re.DOTALL)  ## consider all entities
    return text


### _________________________________________________________

def get_country_demonym(entity):

    country = CountryInfo(entity)
    try:
        country_demonym = [country.demonym()] ## "UK" --> "British", "Canada" --> "Canadian"
    except:
        country_demonym = []
    try:
        country_alt_spelling = country.alt_spellings()  ## "UK" --> "Great Britain
    except:
        country_alt_spelling = []
    finally:
        entity_alt_list = country_demonym + country_alt_spelling

    return entity_alt_list


def load_country_demonyms(config, conflict_entity, load_new = False):

    if Path(config.get_path("entity_retrieval") / "country_demonyms.pkl").exists() and load_new == False:
        country_demonym_dict = load_file(config.get_path("entity_retrieval") / "country_demonyms", ftype="pkl")

    else:
        print("building country_demonyms")
        entity_list = list(conflict_entity.values())
        country_demonym_dict = dict()

        for entity in tqdm(entity_list):
            country_demonyms = get_country_demonym(entity)
            country_demonym_dict[entity] = country_demonyms
        store_file(config.get_path("entity_retrieval") / "country_demonyms", country_demonym_dict, "pkl")

    return country_demonym_dict



if __name__ == "__main__":

    config = configs.ConfigBase()
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    redirect_entity = load_file(config.get_path("entity_retrieval") / "redirect_entity", ftype="pkl")

    redirect_entity_alt = get_country_demonym(entity_dict, redirect_entity)
    redirect_entity = {**redirect_entity_alt, **redirect_entity}

    redirect_entity = store_file(config.get_path("entity_retrieval") / "redirect_entity", redirect_entity, "pkl", "csv")
