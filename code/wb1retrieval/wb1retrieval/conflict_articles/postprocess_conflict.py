from itertools import chain
from xml.etree import cElementTree as ET

from wb0configs import configs
from wb0configs.helpers import store_file, load_file, load_xml, store_xml


def rename_redirected_entities(entity, redirect_entity):

    if entity in redirect_entity.keys():
        return redirect_entity[entity]
    else:
        return entity



def filter_conflict_entity(conflict_entity_name, entity_dict, redirect_entity):
    # conflict_entity_flat_dict = {key: list(chain.from_iterable(val)) for key, val in conflict_entity_name.items()}
    print(f"original conflict_entity_name: {conflict_entity_name.items()}")

    conflict_entity_name_adj = dict() ## conflict_entity_name -->  conflict_entity_name_adj
    conflict_entity_id = dict()

    entity_dict_reverse = {redirect_name: old_name for old_name, redirect_name in entity_dict.items()}
    relevant_entities = list(entity_dict_reverse.keys()) ## all entities allowed
    print(f"all relevant_entities: {relevant_entities}\n")

    for conflict in conflict_entity_name.keys():
        combatants_list_name = list() ## combatants per conflict (pagetitle)
        combatants_list_id = list()  ## combatants per conflict (id)

        for combatant in conflict_entity_name[conflict]:
            combatant_list_name = list() ## combatant from combatants (pagetitle)
            combatant_list_id = list()  ## combatants from combatants (id)

            for entity in combatant:
                entity = rename_redirected_entities(entity, redirect_entity) ## change entity name if redirect

                if entity in relevant_entities: ## check if entity in relevant_entities
                    combatant_list_name.append(entity) ## append name
                    combatant_list_id.append(entity_dict_reverse[entity]) ## append id
                else:
                    print(f"entity '{entity}' not in relevant_entities")

            if (len(combatant_list_name) > 0): ## at least one entity in combatant
                combatants_list_name.append(combatant_list_name)
                combatants_list_id.append(combatant_list_id)

        if len(combatants_list_name) >= 2:  ## at least two combatants per conflict
            conflict_entity_name_adj[conflict] = combatants_list_name
            conflict_entity_id[conflict] = combatants_list_id

    print(f"\nconflict_entity_name old: {len(conflict_entity_name.keys())}, conflict_entity_name_adj new: {len(conflict_entity_name_adj.keys())}")
    print(f"conflicts: \t conflict_entity_name old: {len(conflict_entity_name.keys())}, conflict_entity_name_adj new: {len(conflict_entity_name_adj.keys())}")


    entity_list = list(chain.from_iterable(conflict_entity_name_adj.values())) ## over all conflict
    entity_list = list(set(chain.from_iterable(entity_list))) ## over all belligerents per conflict
    print(f"\nthe difference between relevant_entities and conflict_entity_name_adj is {list(set(relevant_entities) - set(entity_list))}")
    print(f"entity: \t entity_list {len(entity_list)}")

    store_file(config.get_path("conflict_retrieval") / "conflict_entity_name", conflict_entity_name_adj, "csv", "pkl")
    store_file(config.get_path("conflict_retrieval") / "conflict_entity_id", conflict_entity_id, "csv", "pkl")
    #store_file(config.get_path("conflict_retrieval") / "conflict_entity_id", conflict_entity_name_adj, "csv", "pkl")
    return conflict_entity_name_adj



def filter_conflict_xml(xml_root, conflict_entity_name):

    relevant_xml_root = ET.Element("root")  ## all relevant xml pages will be added here

    for page in xml_root.findall('page'):
        pageid = int(page.find('id').text)

        if pageid in conflict_entity_name.keys():  ## check if id in article list
            relevant_xml_root.append(page)

    relevant_xml_tree = ET.ElementTree()
    relevant_xml_tree._setroot(relevant_xml_root)
    store_xml(config.get_path("conflict_retrieval") / "conflict_pages.xml", relevant_xml_tree)
    print("\n --> filter_conflict_xml: done")



def filter_conflict_id_name(conflict_id_name, conflict_entity_name):

    relevant_conflict_id_name = dict()
    for conflict_id in conflict_entity_name.keys():
        relevant_conflict_id_name[conflict_id] = conflict_id_name[conflict_id]

    store_file(config.get_path("conflict_retrieval") / "conflict_id_name", relevant_conflict_id_name, "csv", "pkl")
    print("\n --> filter_conflict_id_name: done")



if __name__ == "__main__":

    config = configs.ConfigBase()
    conflict_entity_dict = load_file(config.get_path("conflict_retrieval") / "pre" / "conflict_entity_name", ftype ="pkl")
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype ="pkl")
    redirect_entity = load_file(config.get_path("entity_retrieval") / "redirect_entity", ftype="pkl")

    conflict_entity_dict = filter_conflict_entity(conflict_entity_dict, entity_dict, redirect_entity)

    ## filter xml files
    xml_tree, xml_root = load_xml(config.get_path("conflict_retrieval") / "pre" / "conflict_pages.xml")
    filter_conflict_xml(xml_root, conflict_entity_dict)

    ## filter conflict_id_name
    conflict_id_name = load_file(config.get_path("conflict_retrieval") / "pre" / "conflict_id_name", ftype="pkl")
    filter_conflict_id_name(conflict_id_name, conflict_entity_dict)

