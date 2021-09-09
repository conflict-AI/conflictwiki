from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, merge_xml, load_xml
from wb1retrieval.parsing.infobox_parsing import extract_infobox_from_xml, separate_infobox
from wb1retrieval.parsing import infobox_parsing
from wb2dataprep.parsing.xml_cleaning import remove_xml_markup

from abc import ABC, abstractmethod
from tqdm import tqdm
import wikitextparser as wtp



class PreprocessParser(ABC):

    def __init__(self, configs):
        self.config  = configs
        self.id_section = dict()
        self.id_info = dict()


    def extract_page_info(self, parsed_page):
        ## function needs to return sectioncontent
        page_info = infobox_parsing.infobox_from_xml(parsed_page)
        return page_info


    def parse_section(self, parsed_page, pageid, pagetitle):

        parsed_sections = dict()
        sections = parsed_page.sections
        n_sections = len(sections)

        if n_sections > 0:
            for i in range(n_sections):  ## iterate sections
                sectiontitle = sections[i].title

                if sectiontitle == None:  ## sometimes start section does not have a title
                    sectiontitle = "Summary"

                sectiontitle = sectiontitle.lstrip().rstrip()

                if sectiontitle.lower() not in ["cited sources","sources","literature","see also", "bibliography", "references", "further reading",
                                        "external links", "notes", "footnotes", "other", "citations","primary sources","secondary sources","tertiary sources"]:  ## remove unnecessary sections

                    sectioncontent = sections[i].contents
                    _, _, end_index = extract_infobox_from_xml(sectioncontent)  ## jump infobox
                    sectioncontent = separate_infobox(sectioncontent, end_index + 2)  ## separate infoxbox

                    sectioncontent = self.pre_cleaning_section(sectioncontent, pageid, pagetitle)
                    sectioncontent = remove_xml_markup(sectioncontent)  ## remove markup
                    sectioncontent = self.post_cleaning_section(sectioncontent, pageid, pagetitle)

                    parsed_sections[sectiontitle] = sectioncontent

        return parsed_sections


    def parse_xml(self, xml_root):

        for page in tqdm(xml_root.findall('page')):
            pageid = int(page.find('id').text)
            pagetitle = page.find('title').text
            pagecontent = page.find('revision').find('text').text
            page_info = self.extract_page_info(pagecontent) ## extract page level info such as info box

            ## convert xml to text
            parsed_page = wtp.parse(pagecontent)  ## parse page

            parsed_sections = self.parse_section(parsed_page, pageid, pagetitle)  ## parse sections
            self.id_section[pageid] = parsed_sections  ## fill content dictionary
            self.id_info[pageid] = page_info           ## fill content info


    @abstractmethod
    def pre_cleaning_section(self, sectioncontent, pageid, pagetitle):
        ## function needs to return sectioncontent
        raise NotImplementedError("pre_process_raw_section is not implemented")


    @abstractmethod
    def post_cleaning_section(self, sectioncontent, pageid, pagetitle):
        ## function needs to return sectioncontent
        raise NotImplementedError("post_cleaning_section is not implemented")

