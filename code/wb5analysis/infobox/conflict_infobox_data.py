from wb0configs import configs
from wb0configs.helpers import store_file, load_file
from wb1retrieval.parsing.infobox_parsing import combatant_extraction

import re
from wikitextparser import remove_markup
import parsedatetime
from collections import defaultdict
import pandas as pd
import itertools
import dateparser

cal = parsedatetime.Calendar()


def build_url_from_title(article_title):
    base_url = "https://en.wikipedia.org/wiki/"
    article_title = article_title.replace(" ", "_")
    return base_url + article_title


def general_info(c_info_dict, c_e_id, conflict_id_name, entity_id_name):

    for c_id, belligerent_ids in c_e_id.items():
        belligerent_names = list()

        for b in belligerent_ids:
            b_names = list()
            for entity_id in b:
                b_names.append(entity_id_name[entity_id])
            belligerent_names.append(b_names)

        c_info_dict[c_id]["conflict_name"] = conflict_id_name[c_id]  ## conflict_name
        c_info_dict[c_id]["url"] = build_url_from_title(conflict_id_name[c_id])
        #c_info_dict[c_id]["belligerent_names"] = belligerent_names  ## belligerent_names
        #c_info_dict[c_id]["belligerent_ids"] = belligerent_ids  ## belligerent_ids
        c_info_dict[c_id]["n_belligerents"] = len(belligerent_ids)  ## number of belligerents
        c_info_dict[c_id]["n_entities"] = len(list(itertools.chain(*belligerent_ids)))  ## number of entities

        general_tags = ["conflict_name", "url", "n_belligerents", "n_entities"] #"belligerent_names", "belligerent_ids"

    return c_info_dict, general_tags



def infobox_info(c_info_dict, id_info):

    info_tags = ["place", "date", "date_start", "date_end", "status", "casualties", "casualties_num", "casualties_sum",
                 "strength", "strength_num", "strength_sum", "commander", "result"]

    for c_id, c_info in id_info.items():

        ## ensure all tags are present for unification
        for t in info_tags:
            if t in ["casualties", "casualties_num", "casualties_sum", "strength", "strength_num", "strength_sum",
                     "commander"]:
                c_info_dict[c_id][t] = dict()
            else:
                c_info_dict[c_id][t] = None

        for k, v in c_info.items():
            if (k in info_tags) or (k[:-1] in info_tags):
                if k[:-1] in ("commander"):
                    v = combatant_extraction(v)
                    # info_dict[c_id][k[:-1]].append((int(k[-1])-1,v))
                    c_info_dict[c_id][k[:-1]][int(k[-1]) - 1] = v

                if k[:-1] in ["casualties", "strength"]:
                    v = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'\1', v,
                               flags=re.IGNORECASE | re.DOTALL)  ## consider [[ | ]]
                    v = re.sub(r'(\[\[)(.+?)(\]\])', r'\2', v,
                               flags=re.IGNORECASE | re.DOTALL)  ## consider [[ ]]
                    v = re.sub(r'<(.+?)/>', '', v)
                    v = remove_markup(v)  ## mediawiki wikitextparser
                    v = re.sub(r'({)(.+?)(})', r'\2: ', v)  ## remove { ... }
                    # info_dict[c_id][k[:-1]].append((int(k[-1])-1,v))
                    c_info_dict[c_id][k[:-1]][int(k[-1]) - 1] = v

                    ## extract numbers
                    v_num = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", v)
                    if len(v_num) > 0:
                        v_num = [num.replace(",", "") for num in v_num]  ## remove commas
                        v_num = [num.replace(".", "") for num in v_num]  ## remove dots
                        v_num = [float(num) for num in v_num]  ## make float
                        v_num = [num for num in v_num if num >= 0]  ## remove negative numbers
                        c_info_dict[c_id][k[:-1] + "_num"][int(k[-1]) - 1] = v_num  ## float numbers
                        c_info_dict[c_id][k[:-1] + "_sum"][int(k[-1]) - 1] = sum(v_num)  ## sum of float numbers

                if k in ["place"]:
                    v = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'\1', v,
                               flags=re.IGNORECASE | re.DOTALL)  ## consider [[ | ]]
                    v = re.sub(r'(\[\[)(.+?)(\]\])', r'\2', v, flags=re.IGNORECASE | re.DOTALL)  ## consider [[ ]]
                    v = re.sub(r'<(.+?)/>', '', v)
                    v = re.sub('\*', '', v)
                    v = re.sub('\|', '; ', v)
                    v = re.sub(r'([{])([.:=]+?)([}])', r'\2', v)  ## remove { ... }
                    v = re.sub(r'(\d)([(])', r'\1 (', v)
                    v = re.sub(r'([(])(.+?)([)])', r'\2', v)  ## remove { ... }
                    v = remove_markup(v)
                    v = re.sub('\\/', '', v)
                    v = re.sub(u'–', '-', v)
                    c_info_dict[c_id][k] = v

                if k in ["date"]:
                    v = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'\1', v,
                               flags=re.IGNORECASE | re.DOTALL)  ## consider [[ | ]]
                    v = re.sub(r'(\[\[)(.+?)(\]\])', r'\2', v, flags=re.IGNORECASE | re.DOTALL)  ## consider [[ ]]
                    v = re.sub(r'<(.+?)/>', '', v)
                    v = re.sub('\*', '', v)
                    v = re.sub('\|', '; ', v)
                    v = re.sub(r'([{])([.:=]+?)([}])', r'\2', v)  ## remove { ... }
                    v = re.sub(r'(\d)([(])', r'\1 (', v)
                    v = re.sub(r'([(])(.+?)([)])', r'\2', v)  ## remove { ... }
                    v = remove_markup(v)
                    v = re.sub('\\/', '', v)
                    v = re.sub(u'–', '-', v)

                    c_info_dict[c_id][k] = v

                    ## split date
                    v_split = v.split("-")

                    if len(v_split[0]) <= 3 and len(v_split) > 1:  ## then number:
                        v_start = v_split[0] + v_split[1][3:]
                        if len(v_split) > 1:
                            v_end = v_split[1]

                    else:  ## two full dates
                        v_start = v_split[0]
                        if len(v_split) > 1:
                            v_end = v_split[1]

                    if len(v_start) > 0:
                        v_start_dt = dateparser.parse(v_start)

                        if v_start_dt == None:
                            v_start_dt = cal.parseDT(v_start)[0]

                    if len(v_end) > 0:
                        v_end_dt = dateparser.parse(v_end)
                        if v_end_dt == None:
                            v_end_dt = cal.parseDT(v_end)[0]

                    if v_start_dt != None and v_end_dt != None and v_end_dt.year < v_start_dt.year:
                        v_start_dt.replace(year=v_end_dt.year)

                    if v_start_dt != None:
                        c_info_dict[c_id][k + "_start"] = v_start_dt.strftime("%Y-%m-%d")

                    if v_end_dt != None:
                        c_info_dict[c_id][k + "_end"] = v_end_dt.strftime("%Y-%m-%d")



                if k in ["result", "status"]:
                    v = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'\1', v,
                               flags=re.IGNORECASE | re.DOTALL)  ## consider [[ | ]]
                    v = re.sub(r'(\[\[)(.+?)(\]\])', r'\2', v, flags=re.IGNORECASE | re.DOTALL)  ## consider [[ ]]
                    v = remove_markup(v)  ## mediawiki wikitextparser
                    v = re.sub(r'<(.+?)/>', '', v)
                    v = re.sub('\|', '; ', v)
                    v = re.sub(r'({)(.+?)(})', r'\2: ', v)  ## remove { ... }
                    v = re.sub('\*', '', v)
                    v = re.sub('\\/', '. ', v)
                    v = re.sub('  ', '. ', v)
                    c_info_dict[c_id][k] = v

    return c_info_dict, info_tags



def store_info_dict(config, c_info_dict, general_tags, info_tags):
    df = pd.DataFrame.from_dict(c_info_dict, orient='index')
    df.index.name = 'conflict_id'
    df = df[general_tags + info_tags]  ## sort columns
    df.to_csv(config.get_path("data_publishing") / "conflict" / "info" /  "conflict_info.csv", index=True)
    store_file(config.get_path("data_publishing") / "conflict" / "info" /  "conflict_info", c_info_dict, "pkl")
    df = df.reset_index()
    df.to_json(config.get_path("data_publishing") / "conflict" / "info" /   "conflict_info.json", index=True, force_ascii=False, lines=True,orient="records")
    print(df)


if __name__ == "__main__":

    config = configs.ConfigBase()
    id_info = load_file(config.get_path("conflict_dataprep") / "id_info", ftype ="pkl")

    c_e_id = load_file(config.get_path("conflict_retrieval") / "conflict_entity_id", ftype="pkl")
    conflict_id_name = load_file(config.get_path("conflict_retrieval") / "conflict_id_name", ftype="pkl")
    entity_id_name = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")

    ###
    c_info_dict = defaultdict(dict)
    c_info_dict, general_tags = general_info(c_info_dict, c_e_id, conflict_id_name, entity_id_name)
    c_info_dict, info_tags = infobox_info(c_info_dict, id_info)
    store_info_dict(config, c_info_dict, general_tags, info_tags)