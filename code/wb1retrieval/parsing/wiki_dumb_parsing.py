import itertools
import bz2
from abc import ABC, abstractmethod
from xml.etree import cElementTree as ET
from tqdm import tqdm



class WikiDumbParser(ABC):

    def __init__(self, configs):
        self.config  = configs
        self.relevant_xml_root = ET.Element("root")  ## all relevant xml pages will be added here
        self.relevant_xml_tree = ET.ElementTree()


    def get_xmlpage_streambatch(self, offset_1 = int(617), offset_2 = int(99999)):
        """
        :param art_list:
        :param relevant_art_xml_root:
        :param offset_1:
        :param offset_2:
        :return:
        """

        #stream_file = open(config.get_path("wiki_stream"), "rb") ## main stream file
        #stream_file.close()


        with open(self.config.get_path("wiki_stream"), "rb") as stream_file: ## main stream file

            decompressor = bz2.BZ2Decompressor()

            stream_file.seek(offset_1)
            content = stream_file.read(offset_2 - offset_1)

            content = decompressor.decompress(content)  ## bz2 stream
            content = content.decode(encoding='utf-8')  ## bytes to string

            xml_content = itertools.chain('<root>', content, '</root>')
            root = ET.fromstringlist(xml_content)

        return root



    def parse_index_file(self, candidate_list: list, based_on: str = "pageid"):
        """
        parse index file and get actual contents from stream file
        :param relevant_art_xml_root:
        :param candidate_list:
        :return:
        """

        self.candidate_list = candidate_list
        batch_candidate_list = list()
        currrent_start_offset = int(617)

        with bz2.open(self.config.get_path("wiki_index"), "rt") as index_file:

            for line in tqdm(index_file):
                line = line.rstrip('\n')
                line = line.split(':')  # offset, pageid, title

                offset = int(line[0])
                pageid = int(line[1])
                pagetitle = str(line[2])

                if offset != currrent_start_offset: ## (1) switch offset after article chunk

                    if len(batch_candidate_list) > 0:  ## check if art_list contains any entries to save computation
                        xmlpages_root = self.get_xmlpage_streambatch(currrent_start_offset, offset)
                        self.select_relevant_pages(batch_candidate_list, xmlpages_root)

                    currrent_start_offset = offset  ## reset offset
                    batch_candidate_list = list()  ## reset list


                if based_on == "pageid":
                    if pageid in self.candidate_list: ## (2) check if id is in conflict candidate_list
                        #self.candidate_list.remove(pageid)
                        batch_candidate_list.append(pageid)  ## append page ids

                elif based_on == "pagetitle":
                    if pagetitle in self.candidate_list: ## (2) check if pagetitle is in candidate_list
                        #self.candidate_list.remove(pagetitle)
                        batch_candidate_list.append(pageid)  ## append page ids

        ## create new xml file
        print(f"number of relevant pages: {len(self.relevant_xml_root.findall('page'))}")
        self.relevant_xml_tree._setroot(self.relevant_xml_root)



    @abstractmethod
    def select_relevant_pages(self,relevant_id_list, xmlpages_root):
        ## methods needs to append selected xml pages to self.relevant_xml_root
        raise NotImplementedError("select_relevant_pages is not implemented")