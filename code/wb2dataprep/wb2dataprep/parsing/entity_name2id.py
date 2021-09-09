from wb0configs import configs
from wb0configs.helpers import store_file, load_file

import re


def change_entity_name2id(text, entity_dict):
    entity_dict_r = {v: k for k,v in entity_dict.items()}
    text = re.sub(r"(<)(.+?)(/>)", lambda m: '<{}/>'.format(entity_dict_r.get(m.group(2))), text)  ## consider entities in [[ ]]
    return text


if __name__ == "__main__":
    config = configs.ConfigBase()

    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    text = "bla bla <Afghanistan/>, <unkowen/>"

    text = change_entity_name2id(text, entity_dict)
    print(text)