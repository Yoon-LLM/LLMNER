"""
    Train data should be in this format

    ```
    Redact the names of patients, doctors, and usernames from the text of the report.

    e.g.) 
    INPUT TEXT:
    <TEXT>
    ~~~~~~~~
    </TEXT>


    REDACTED TEXT:
    <TEXT>
    ~~~~~~~
    </TEXT>
    ```
"""

import json, re, glob
import xml.etree.ElementTree as ET


TAG_dict = {
    "NAMES":  ['PATIENT', 'DOCTOR', 'USERNAME'],
    "DATE": ['DATE'],
    "AGE": ['AGE'],
    "LOCATION": ['ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP'],
    "CONTACT": ['PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR'],
    "IDTAG": ['SSN','MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM'],
    "PROFESSION": ['PROFESSION'],
    "ALL": ['PATIENT', 'DOCTOR', 'USERNAME', 'DATE', 'AGE', 'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET', 'CITY', 'STATE', 'COUNTRY', 'ZIP', 'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR', 'SSN','MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT', 'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM', 'PROFESSION']
}
Instruction_dict = {
    "NAMES" : "Replace any names, acronyms, or initials that could identify individual people with '[redacted]'.",
    "DATE" : "Replace any date informations with '[redacted]'.",
    "AGE" : "Replace any strings that look like age such as 'something year old' or 'age 37' with '[redacted]'",
    "LOCATION" : "Replace any location details like room numbers, streets, cities, states, zip codes, and countries with '[redacted]'.",
    "CONTACT": "Replace any contact information with '[redacted]'.",
    "PROFESSION": "Replace professions such as “manages” with '[redacted]'.",
    "IDTAG" : "Replace any IDs with '[redacted]'.",
    "ALL": """Prompt:
Task: Please anonymize the following clinical note. 
Specific Rules: Replace all the following information with the term “[redacted]”: 
    Redact any strings that might be a name or acronym or initials, patients’ names, doctors’ names, the names of the M.D. or Dr.,
    Redact any pager names, medical staff names,
    Redact any strings that might be a location or address, such as “3970 Longview Drive”, 
    Redact any strings that look like “something years old” or “age 37”, 
    Redact any dates and IDs and numbers and record dates, ID-like strings
    Redact clinic and hospital names, 
    Redact professions such as “manager”, 
    Redact any contact information: 
"""

}

TAGS = ["NAMES", "DATE", "AGE", "LOCATION", "CONTACT", "IDTAG"]

class BaseXMLReader:
    def __init__(self, xml_file) -> None:
        self.xml_file = xml_file
        self.text = self.extract_text(xml_file)

    def extract_text(self, xml_file):
        tree = ET.parse(xml_file)
        text = tree.find("TEXT").text
        return text

class Redactor:
    def __init__(self, xml_file, tag_types=['PATIENT', 'DOCTOR', 'USERNAME']) -> None:
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.tags = self.extract_tags(xml_file)
        self.text = self.extract_text(xml_file)
        self.tag_types = tag_types
        self.tag_list = [tag for tag in self.tags if tag['TYPE'] in tag_types]

    def extract_text(self, xml_file):
        text = self.tree.find("TEXT").text
        return text

    def extract_tags(self, xml_file):
        tags = self.tree.find("TAGS")
        tag_list = []
        for tag in tags:
            tag_dict = {'tag':tag.tag}
            tag_dict.update(tag.attrib) 
            tag_list.append(tag_dict)
        return tag_list
    
    def add_tags(self, tag_list):
        tags = self.tree.find("TAGS")
        for tag in tag_list:
            tags.append(tag)
        self.tree

    @staticmethod           
    def redact_tag(text, tag):
        start = int(tag['start'])
        end = int(tag['end'])
        redacted_text = text[:start] + '*' *len(tag['text']) + text[end:]
        return redacted_text

    def redacted_text(self, redact_with = None):
        text = self.text
        for tag in self.tag_list:
            text = self.redact_tag(text, tag)
        if redact_with != None:
            text = self.redact_with(text, redact_with)

        return text

    @staticmethod
    def custom_split(text):
        return [item for item in re.split(r'\*{2,}', text) if item != '']


    def redact_with(self, redacted_text, redact_with):
        split_text = self.custom_split(redacted_text)
        text = redact_with.join(split_text)
        return text
    
if __name__ =="__main__":
    json_list = []
    TAGS = ["ALL"]
    for TAG in TAGS: #IDTAG # LOCATION #NAMES #DATE #AGE
        xml_list = glob.glob("G:\\내 드라이브\\data\\NERdata\\*.xml", recursive=True)
        for xml in xml_list:
            redactor = Redactor(xml, tag_types=TAG_dict[TAG]) 
            json_list.append({
                'instruction': Instruction_dict[TAG],
                'input':redactor.text,
                'output':redactor.redacted_text(redact_with="[redacted]")
            })
            # redactor.redacted_text(redact_with="[redacted]")

        with open(f"{TAG}_deid_file.json", "w") as json_file:

            json.dump(json_list, json_file)