from wb0configs import configs
from wb0configs.helpers import store_file, load_file, load_xml
from wb2dataprep.parsing.preprocess_parsing import PreprocessParser
from wb2dataprep.parsing.coreference_resolution import country_demonym_resolution, load_country_demonyms, regex_entity_coreference
from wb2dataprep.parsing.entity_name2id import change_entity_name2id

import re
from itertools import chain



def find_relevent_entity_links(text, entity_list, redirect_entity):

    for entity in entity_list:
        text = re.sub(rf"(\[\[)({entity})(\]\])", r'<\2/>', text, flags=re.IGNORECASE)  ## consider entities in [[ ]]

        if entity in redirect_entity.values(): ## consider redirect entities in [[ ]]
            relevant_redirect_entity = list(set(re.findall(r"(?=(" + '|'.join(list(redirect_entity.keys())) + r"))", text)))

            if len(relevant_redirect_entity) > 0:
                for re_entity in relevant_redirect_entity:
                    text = re.sub(rf"(\[\[)({re_entity})(\]\])", rf"<{redirect_entity[re_entity]}/>", text, flags=re.IGNORECASE) ## consider entities in [[ ]]
    return text



def prepare_entity_links(text):
    ## change [[ | ]] to [[ ]]

    text = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'[[\1]]', text, flags= re.IGNORECASE | re.DOTALL) ## consider [[ | ]]
    text = re.sub(r'(\[\[)(.+?)(\]\])', r'[[\2]]', text, flags= re.IGNORECASE | re.DOTALL) ## consider [[ ]]
    #text = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'<\1/>', text) ## consider [[ | ]]
    #text = re.sub(r'(\[\[)(.+?)(\]\])', r'<\2/>', text) ## consider [[ ]]

    return text




class ConflictPreprocessParser(PreprocessParser):

    def __init__(self,config, conflict_entity, entity_dict, redirect_entity):
        super().__init__(config)
        self.conflict_entity = conflict_entity
        self.entity_dict = entity_dict
        self.redirect_entity = redirect_entity
        self.country_demonyms = load_country_demonyms(self.config, self.entity_dict, load_new = True)


    def pre_cleaning_section(self, sectioncontent, pageid, pagetitle):

        entity_list = list(chain(*self.conflict_entity[pageid]))
        sectioncontent = prepare_entity_links(sectioncontent) ## switch from [[]] to < />
        sectioncontent = find_relevent_entity_links(sectioncontent, entity_list, self.redirect_entity) ## keep all < /> which are relevant entities
        return sectioncontent

    def post_cleaning_section(self, sectioncontent, pageid, pagetitle): ## coreference resolution

        entity_list = list(chain(*self.conflict_entity[pageid]))
        sectioncontent = regex_entity_coreference(sectioncontent, entity_list, redirect_entity)
        sectioncontent = country_demonym_resolution(sectioncontent, entity_list, self.country_demonyms)
        #sectioncontent = spacy_entity_coreference(sectioncontent, entity_list)

        ## name to id
        sectioncontent = change_entity_name2id(sectioncontent, self.entity_dict)
        return sectioncontent







if __name__ == "__main__":

    config = configs.ConfigBase()
    xml_tree, xml_root = load_xml(config.get_path("conflict_retrieval") / "conflict_pages.xml")
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    redirect_entity = load_file(config.get_path("entity_retrieval") / "redirect_entity", ftype="pkl")
    conflict_entity_name = load_file(config.get_path("conflict_retrieval") / "conflict_entity_name", ftype="pkl")

    parser = ConflictPreprocessParser(config, conflict_entity_name, entity_dict, redirect_entity)
    parser.parse_xml(xml_root)
    store_file(config.get_path("conflict_dataprep") / "id_section", parser.id_section, "csv", "pkl")
    store_file(config.get_path("conflict_dataprep") / "id_info", parser.id_info, "csv", "pkl")

    #store_xml(config.get_path("conflict_retrieval") / "conflict_pages.xml", parser.relevant_xml_tree)
    #store_file(config.get_path("conflict_retrieval") / "pre" / "conflict_entity_name", parser.conflict_entity_name, "csv", "pkl")
