### browsing all articles in wiki category

import wikipediaapi
from pathlib import Path
import re

from wb0configs import configs
from wb0configs.helpers import store_file


def parse_categorymembers(categorymembers, art_names: dict, level=0, max_level=3):
    for c in categorymembers.values():
        #print("%s: %s ,pageid: %i , (ns: %d)" % ("*" * (level + 1), c.title, c.pageid, c.ns))

        if "Category:" not in c.title: ## exclude categories
            art_names[int(c.pageid)] = str(c.title)

        if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            parse_categorymembers(c.categorymembers, art_names, level=level + 1, max_level=max_level)
    return art_names


if __name__ == "__main__":

    config = configs.ConfigBase()
    wiki_wiki = wikipediaapi.Wikipedia('en')

    file_name = "conflicts_2001-2021"
    max_level = 3

    print(f"Category members: {file_name}, max_level: {max_level}")
    art_names = dict() ## list to be filled

    for cat_name in range(2021, 2000, -1):
        cat = wiki_wiki.page("Category:Conflicts in " + str(cat_name))
        art_names = parse_categorymembers(cat.categorymembers, art_names, level=0, max_level=max_level)

        print(f"{cat_name} – number of articles: {len(art_names.keys())} – {art_names.keys()}")

        #cat_name = re.sub("[^0-9a-zA-Z]+", "_", cat_name.lower()) + "_max_level_" + str(max_level)
    store_file(Path(config.get_path("conflict_retrieval")) / file_name, art_names, "csv", "pkl")


