from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb1retrieval.parsing.infobox_parsing import combatant_extraction

import re
from wikitextparser import remove_markup
import parsedatetime
from collections import defaultdict
import pandas as pd
import itertools

cal = parsedatetime.Calendar()


def build_url_from_title(article_title):
    base_url = "https://en.wikipedia.org/wiki/"
    article_title = article_title.replace(" ", "_")
    return base_url + article_title


def bracket_list_extraction(data):

    data = re.sub(r'\{\{cite(.+?)\}\}', '', data, flags=re.IGNORECASE)
    data = re.sub(r'\[\[Image:(.+?)(?:\||[^a-zA-Z\d\s\'.-])', '', data, flags=re.IGNORECASE)
    data = re.sub(r'\[\[File:(.+?)(?:\||[^a-zA-Z\d\s\'.-])', '', data, flags=re.IGNORECASE)

    flag_entities = re.findall(r'\{\{(?:f|F)lag\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags=re.IGNORECASE | re.DOTALL)
    flagicon_entities = re.findall(r'\{\{(?:f|F)lagicon\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags=re.IGNORECASE | re.DOTALL)
    flagdeco_entities = re.findall(r'\{\{(?:f|F)lagdeco\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags=re.IGNORECASE | re.DOTALL)
    flagu_entities = re.findall(r'\{\{(?:f|F)lagu\|(.+?)[^a-zA-Z\d\s:\'.-]', data, flags=re.IGNORECASE | re.DOTALL)
    flagcountry_entities = re.findall(r'\{\{(?:f|F)lagcountry\|(.+?)[^a-zA-Z\d\s\'.-]', data,
                                      flags=re.IGNORECASE | re.DOTALL)

    bracket_entities_1 = re.findall(r'\}\} \[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags=re.IGNORECASE | re.DOTALL)
    bracket_entities_2 = re.findall(r'\]\] \[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags=re.IGNORECASE | re.DOTALL)
    bracket_entities_3 = re.findall(r'\[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags=re.IGNORECASE | re.DOTALL)

    after_vert_entities = re.findall(r'\|([^\|;=,.*[[]+?)\]\]', data, flags=re.IGNORECASE | re.DOTALL)
    entities = flag_entities + flagicon_entities + flagdeco_entities + flagu_entities + flagcountry_entities + bracket_entities_1 + bracket_entities_2 + bracket_entities_3 + after_vert_entities

    entities = [e.rstrip().lstrip() for e in set(entities) if len(e) >= 3]  ## at least 3 char long
    entities = [e.replace("#", "") for e in set(entities) if len(e) >= 3]  ## at least 3 char long
    return entities



def general_info(e_info_dict, c_e_id, conflict_id_name, entity_id_name):

    general_tags = ["entity_name", "url", "num_conflicts"]  # "conflict_names","conflict_ids"

    for c_id, belligerent_ids in c_e_id.items():

        # belligerents = list(itertools.chain(*belligerent_ids))
        for i, b in enumerate(belligerent_ids):
            for entity_id in b:
                if entity_id in e_info_dict:
                    # e_info_dict[entity_id]["conflict_names"].append(conflict_id_name[c_id])
                    # e_info_dict[entity_id]["conflict_ids"].append(c_id)
                    e_info_dict[entity_id]["num_conflicts"] += 1
                else:
                    e_info_dict[entity_id]["entity_name"] = entity_id_name[entity_id]
                    e_info_dict[entity_id]["url"] = build_url_from_title(entity_id_name[entity_id])
                    # e_info_dict[entity_id]["conflict_names"] = [conflict_id_name[c_id]]
                    # e_info_dict[entity_id]["conflict_ids"] = [c_id]
                    e_info_dict[entity_id]["num_conflicts"] = 1

    return e_info_dict, general_tags



def infobox_info(e_info_dict, id_info):

    info_tags = ["iso", "language", "ideology", "religion"]

    for e_id, e_info in id_info.items():

        ## check if entity exists in conflict
        if e_id in e_info_dict:

            ## ensure all tags are present for unification
            for t in info_tags:
                e_info_dict[e_id][t] = list()

            for k, v in e_info.items():
                # print(e_info_dict[e_id]["entity_name"], k, v[:20])

                if k[:8].lower() in ["ideology"]:
                    v = bracket_list_extraction(v)
                    if len(v) > 0:
                        e_info_dict[e_id]["ideology"] += v
                if k[:8].lower() in ["language"] or k[:13].lower() in ["official_lang"]:
                    v = bracket_list_extraction(v)
                    if len(v) > 0:
                        e_info_dict[e_id]["language"] += v
                if k[:8].lower() in ["religion"]:
                    v = bracket_list_extraction(v)
                    if len(v) > 0:
                        e_info_dict[e_id]["religion"] += v
                if k[:3].lower() in ["iso"]:
                    v_list = re.findall(r'\|([^\|;=,.*[[]+?)\]\]', v, flags=re.IGNORECASE | re.DOTALL)
                    if len(v_list) > 0:
                        v = v_list[0].split("-")[0]
                    else:
                        v = v.split("-")[0]
                    if len(v) == 2 and v.isalpha():
                        e_info_dict[e_id]["iso"] = v

                elif k.lower() in ["currency_code"]:  ## ISO:
                    e_info_dict[e_id]["iso"] = v[:2]

    return e_info_dict, info_tags



def store_info_dict(config, c_info_dict, general_tags, info_tags):
    df = pd.DataFrame.from_dict(c_info_dict, orient='index')
    df.index.name = 'entity_id'
    df = df[general_tags + info_tags]  ## sort columns
    df.to_csv(config.get_path("data_publishing") / "entity" / "info" /  "entity_info.csv", index=True)
    store_file(config.get_path("data_publishing") / "entity" / "info" / "entity_info", e_info_dict, "pkl")
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "entity" / "info" / "entity_info.json", index=True, force_ascii=False, lines=True,orient="records")
    print(df)


if __name__ == "__main__":

    config = configs.ConfigBase()
    id_info = load_file(config.get_path("entity_dataprep") / "id_info", ftype="pkl")

    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    conflict_id_name = load_file(config.get_path("conflict_retrieval") / "conflict_id_name", ftype="pkl")
    entity_id_name = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")

    ###
    e_info_dict = defaultdict(dict)
    e_info_dict, general_tags = general_info(e_info_dict, c_e_id, conflict_id_name, entity_id_name)
    e_info_dict, info_tags = infobox_info(e_info_dict, id_info)
    store_info_dict(config, e_info_dict, general_tags, info_tags)