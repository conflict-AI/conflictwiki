from itertools import chain
import re

from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, merge_xml
from wb1retrieval.parsing.wiki_dumb_parsing import WikiDumbParser


def check_entity_page(page, pagetitle):

    page_content = page.find('revision').find('text').text

    if page.find('redirect') != None: ## redirect page
        new_pagetitle = re.findall(r'\#REDIRECT \[\[(.+?)(?:\||[^a-zA-Z\d\s\'-])', page_content, flags= re.IGNORECASE | re.DOTALL)

        if len(new_pagetitle) > 0:
           return new_pagetitle[0]  ## first match --> str
        else:
            return None
    else:                              ## page exists
        if len(page_content) > 100:
            return pagetitle
        else:
            return None



class EntityDumbParser(WikiDumbParser):

    def __init__(self,config):
        super().__init__(config)
        self.relevant_entity = dict()
        self.redirect_entity = dict()

    def select_relevant_pages(self, id_list, xmlpages_root):

        for page in xmlpages_root.findall('page'):
            pageid = int(page.find('id').text)
            pagetitle = page.find('title').text

            if (pageid in id_list) and (pagetitle in self.candidate_list):  ## check if id in article list
                self.candidate_list.remove(pagetitle)

                pagetitle_verified = check_entity_page(page, pagetitle)

                if pagetitle_verified == pagetitle: ## proper page with contents
                    print(pageid, pagetitle_verified)
                    self.relevant_entity[pageid] = pagetitle_verified
                    self.relevant_xml_root.append(page)

                elif pagetitle_verified != None: ## remap entity name if redirect
                    self.redirect_entity[pagetitle] = pagetitle_verified

                else:
                    ## drop entity entirely because neither proper page nor redirect
                    pass



if __name__ == "__main__":

    config = configs.ConfigBase()

    conflict_entity_dict = load_file(config.get_path("conflict_retrieval") / "pre" / "conflict_entity_name", ftype ="pkl")
    entity_list = list(conflict_entity_dict.values())

    entity_list = list(chain.from_iterable(entity_list)) ## over all conflict
    entity_list = list(set(chain.from_iterable(entity_list))) ## over all belligerents per conflict

    ## parse first time
    print(f"(1) first pass: entity_list length: {len(entity_list)}, {entity_list}")
    parser = EntityDumbParser(config)
    parser.parse_index_file(candidate_list= entity_list, based_on = "pagetitle")
    relevant_entity = parser.relevant_entity
    xml_tree1 = parser.relevant_xml_tree

    ## prepare second pass on redirects
    entity_list_redirect = list(parser.redirect_entity.values()) ## get redirects
    store_file(config.get_path("entity_retrieval") / "redirect_entity", parser.redirect_entity, "csv", "pkl")
    entity_list_redirect = set(entity_list_redirect) - set(list(relevant_entity.values())) ## some redirects may lead to known entities

    ## parse second time for redirects
    print(f"\n\n(2) second pass: entity_list_redirect length: {len(entity_list_redirect)}, {entity_list_redirect}")
    parser = EntityDumbParser(config)
    parser.parse_index_file(candidate_list= entity_list_redirect, based_on = "pagetitle")
    redirect_relevant_entity = parser.relevant_entity
    xml_tree2 = parser.relevant_xml_tree

    ## merge xml files
    xml_tree = merge_xml(xml_tree1, xml_tree2)
    store_xml(config.get_path("entity_retrieval") / "entity_pages.xml", xml_tree)
    relevant_entity_complete = {**relevant_entity, **redirect_relevant_entity} ## store final, complete relevant entities
    store_file(config.get_path("entity_retrieval") / "entity_id_name", relevant_entity_complete, "csv", "pkl")

