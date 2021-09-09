from wb0configs import configs

import re
from wikitextparser import remove_markup


def entity_brackets_to_links(text):
    text = re.sub(r'\[\[([a-zA-Z\d\s\'\.-]+)\|([a-zA-Z\d\s\'\.-]+)\]\]', r'<\1/>', text)  ## consider [[ | ]]
    text = re.sub(r'(\[\[)(.+?)(\]\])', r'<\2/>', text)  ## consider [[ ]]
    return text


def remove_xml_markup(text):

    text = remove_markup(text) ## mediawiki wikitextparser

    text = text.replace("\n", "") ## own post-processing
    text = text.replace("\n\n", "")
    text = text.replace("<br />", "")
    text = text.replace("<br>", "")
    text = text.replace("=thumb |","")
    text = text.replace("thumb|", "")
    text = text.replace("thumb|upright=0.75|", "")
    text = text.replace("=thumb | upright |","")
    text = re.sub(r'<cite(.+?)/>', '', text)
    text = re.sub(r'<Image(.+?)/>', '', text)
    text = re.sub(r'<File(.+?)/>', '', text)
    text = re.sub(r'<(.+?)/>', '', text)
    text = re.sub(r'(===)(.+?)(===)', r'\2: ', text)  ## remove subtitle sections
    text = re.sub(r'({)(.+?)(})', r'\2: ', text)  ## remove { ... }

    #text = re.sub(r'\[\[(.+?)\]\]', r'', text)  ## remove leftover [] links
    #text = re.sub(r'\[(.+?)\]', r'', text)  ## remove leftover [[]]links
    return text


if __name__ == "__main__":

    config = configs.ConfigBase()
    text = "Operation Enduring Freedom''' ('''OEF''') was the official name used by the [[U.S. government]] for the [[Global War on Terrorism]]. On 7 October 2001, in response to the [[September 11 attacks]], [[President of the United States|President]] [[George W. Bush]] announced that airstrikes targeting [[Al Qaeda]] and the <Taliban/>"

    print(text, "\n")
    text_resolved = entity_brackets_to_links(text)
    print(remove_xml_markup(text_resolved))