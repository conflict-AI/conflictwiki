import re


def combatant_extraction(data):
    #data = data.replace("Image:","")
    #data = data.replace("File:", "")
    # data = data.replace('&amp;amp;', '&')

    data = re.sub(r'\{\{cite(.+?)\}\}', '', data,flags=re.IGNORECASE)
    data = re.sub(r'\[\[Image:(.+?)(?:\||[^a-zA-Z\d\s\'.-])', '', data,flags=re.IGNORECASE)
    data = re.sub(r'\[\[File:(.+?)(?:\||[^a-zA-Z\d\s\'.-])', '', data,flags=re.IGNORECASE)

    flag_entities = re.findall(r'\{\{(?:f|F)lag\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags= re.IGNORECASE | re.DOTALL)
    flagicon_entities = re.findall(r'\{\{(?:f|F)lagicon\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags= re.IGNORECASE | re.DOTALL)
    flagdeco_entities = re.findall(r'\{\{(?:f|F)lagdeco\|(.+?)[^a-zA-Z\d\s\'.-]', data, flags= re.IGNORECASE | re.DOTALL)
    flagu_entities = re.findall(r'\{\{(?:f|F)lagu\|(.+?)[^a-zA-Z\d\s:\'.-]', data, flags= re.IGNORECASE | re.DOTALL)
    flagcountry_entities = re.findall(r'\{\{(?:f|F)lagcountry\|(.+?)[^a-zA-Z\d\s\'.-]', data,  flags= re.IGNORECASE | re.DOTALL)

    #bracket_entities_1 = re.findall(r'\}\} \[\[(.+?)[^a-zA-Z\d\s:-]', data, re.DOTALL)
    #bracket_entities_2 = re.findall(r'\]\] \[\[(.+?)[^a-zA-Z\d\s:-]', data, re.DOTALL)
    bracket_entities_1 = re.findall(r'\}\} \[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags= re.IGNORECASE | re.DOTALL)
    bracket_entities_2 = re.findall(r'\]\] \[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags= re.IGNORECASE | re.DOTALL)
    bracket_entities_3 = re.findall(r'\[\[(.+?)(?:\||[^a-zA-Z\d\s\'.-])', data, flags= re.IGNORECASE | re.DOTALL)
    entities = flag_entities + flagicon_entities + flagdeco_entities +  flagu_entities + flagcountry_entities+ bracket_entities_1 + bracket_entities_2 + bracket_entities_3

    entities = [e.rstrip().lstrip() for e in set(entities) if len(e) >= 3] ## at least 3 char long
    return entities


def unmatched_bracket(text):
    """
    Returns true if there is an unmatched bracket
        this is a sentence {with a bracket } - false
        this is a sentence {with a bracket } and {this - true
    """
    for c in reversed(text):
        if c == "}":
            return False
        elif c == "{":
            return True


def separate_infobox(page_text, end_index):
    return page_text[end_index:]


def extract_infobox_from_xml(page_text):

    start_index = page_text.find("nfobox") - 1
    if start_index == -1 or len(page_text) <= 1:
        #print("No infobox")
        return None, 0, 0

    else:
        bracket_count = 0
        end_index = start_index
        for i in range(start_index, len(page_text)):
            try:
                char = page_text[i]
            except:
                print(i)
                print(page_text)
            if char == "}":
                bracket_count -= 1
            elif char == "{":
                bracket_count += 1

            if bracket_count == -2:
                # reached end of info box
                end_index = i - 1
                break

        infobox_content_list = page_text[start_index:end_index].splitlines()
        return infobox_content_list, start_index, end_index



def infobox_from_xml(page_text, pagetype = "conflict"):

    infobox = {}
    infobox_content_list, start_index, end_index = extract_infobox_from_xml(page_text)

    if infobox_content_list != None:
        if len(infobox_content_list) > 0:
            if len(infobox_content_list[0].split("|", 1)) == 2:
                # need to add pipe back in after the split
                # pipe is needed later on
                infobox_content_list[0] = "|" + infobox_content_list[0].split("|")[1]
            else:
                infobox_content_list = infobox_content_list[1:]

            infobox_merged_content = []
            for line in infobox_content_list:
                line = line.strip()
                if len(line) == 0:
                    continue
                elif line[0] == "|":
                    infobox_merged_content.append(line)
                else:
                    if len(infobox_merged_content) > 0:
                        infobox_merged_content[-1] += " " + line
                    else:
                        infobox_merged_content.append(line)

            for entry in infobox_merged_content:
                key_data = entry.strip().split("=", 1)  # only split on first "="

                # multiple strips because it might look like
                # "    | Key = Text"
                # Need to remove up to |, remove bar, and then strip again
                key = key_data[0].lstrip()[1:].strip()  # removes "|"

                if len(key) > 0 and key[0] == "|":
                    # sometimes have duplicate |'s because why would this be easy
                    # eg: ||NotParticipating=Stewart and Fortas
                    key = key[1:]

                data = ""

                if len(infobox) > 0 and unmatched_bracket(infobox[list(infobox.keys())[-1]]):
                    infobox[list(infobox.keys())[-1]] += " " + key

                if len(key_data) == 2:
                    data = key_data[1].strip()

                infobox[key] = data

    return infobox
