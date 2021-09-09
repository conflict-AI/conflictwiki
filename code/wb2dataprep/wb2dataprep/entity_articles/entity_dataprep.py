from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml
from wb2dataprep.parsing.preprocess_parsing import PreprocessParser





class EntityPreprocessParser(PreprocessParser):

    def __init__(self,config):
        super().__init__(config)
        self.conflict_entity = dict()

    def pre_cleaning_section(self, sectioncontent, pageid, pagetitle):
        pass
        return sectioncontent

    def post_cleaning_section(self, sectioncontent, pageid, pagetitle):
        pass
        return sectioncontent




if __name__ == "__main__":

    config = configs.ConfigBase()

    xml_tree, xml_root = load_xml(config.get_path("entity_retrieval") / "entity_pages.xml")

    parser = EntityPreprocessParser(config)
    parser.parse_xml(xml_root)
    store_file(config.get_path("entity_dataprep") / "id_section", parser.id_section, "csv", "pkl")
    store_file(config.get_path("entity_dataprep") / "id_info", parser.id_info, "csv", "pkl")


    #store_xml(config.get_path("conflict_retrieval") / "conflict_pages.xml", parser.relevant_xml_tree)
    #store_file(config.get_path("conflict_retrieval") / "pre" / "conflict_entity_name", parser.conflict_entity_name, "csv", "pkl")
