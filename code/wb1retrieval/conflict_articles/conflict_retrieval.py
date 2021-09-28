from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml
from wb1retrieval.parsing.wiki_dumb_parsing import WikiDumbParser
from wb1retrieval.parsing import infobox_parsing


def check_infobox_combatant(page):

    page_content = page.find('revision').find('text').text
    # print(page.find('title').text)
    #if "combatant1" in page_content and "combatant2" in page_content:  ## check if belligerents in infobox
    infobox = infobox_parsing.infobox_from_xml(page_content)
    combatants = [val for key, val in infobox.items() if key[:-1] == "combatant"]
    combatants = [infobox_parsing.combatant_extraction(combatant) for combatant in combatants if (len(combatant) > 0)]

    if len(combatants) >= 2: ## at least two combatants / belligerents
        return combatants
    else:
        return None



class ConflictDumbParser(WikiDumbParser):

    def __init__(self,config):
        super().__init__(config)
        self.conflict_entity = dict()
        self.conflict_id_name = dict()

    def select_relevant_pages(self, id_list, xmlpages_root):

        for page in xmlpages_root.findall('page'):
            pageid = int(page.find('id').text)
            pagetitle = str(page.find('title').text)

            if pageid in id_list:  ## check if id in article list
                self.candidate_list.remove(pageid)
                entities = check_infobox_combatant(page)

                if entities != None:
                    print(pageid, page.find('title').text, entities)
                    self.relevant_xml_root.append(page)
                    self.conflict_entity[pageid] = entities
                    self.conflict_id_name[pageid] = pagetitle



if __name__ == "__main__":

    config = configs.ConfigBase()
    cat_file_name = "conflicts_2001-2021"

    conflict_id = load_file(config.get_path("conflict_retrieval") / cat_file_name, ftype ="pkl")

    ## parse index file
    print(f"conflict_id candidates length: {len(conflict_id.items())}, {conflict_id.items()}")

    parser = ConflictDumbParser(config)
    parser.parse_index_file(candidate_list= list(conflict_id.keys()), based_on = "pageid")

    store_xml(config.get_path("conflict_retrieval") / "pre" / "conflict_pages.xml", parser.relevant_xml_tree)
    store_file(config.get_path("conflict_retrieval") / "pre" / "conflict_entity_name", parser.conflict_entity, "csv", "pkl")
    store_file(config.get_path("conflict_retrieval") / "pre" / "conflict_id_name", parser.conflict_id_name, "csv", "pkl")

