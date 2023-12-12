from transformers import pipeline
from preprocess import Redactor
import fire
from glob import glob
import doctest

def get_tag_info(text, ner_pipeline, tags, prompt=None, **kwargs):
    """
    text : str 
    ner_pipeline : pipeline
    return : tag_info : str
    """
    
    if prompt:
        tag_info = ner_pipeline(text) # [{'entity': 'I-PER', 'score': 0.99893445, 'index': 6, 'word': 'John', 'start': 11, 'end': 15}, {'entity': 'I-PER', 'score': 0.99947304, 'index': 7, 'word': 'Smith', 'start': 16, 'end': 21}]
    else:
        tag_info = ner_pipeline(text, prompt)
    return [tag for tag in tag_info if tag['entity'] in tags]

def main(test_data_path, output_path, model_path, **kwargs):
    ner_pipeline = pipeline(model_path)
    for xml in glob(test_data_path+"*", recursive=False):
        redctor = Redactor(xml, tag_types=['I-PER'])
        outputs = get_tag_info(redactor.text, ner_pipeline, tags=['I-PER'])
        
#         redacted_text = redactor.redacted_text(redact_with="[redacted]")
#         for output in outputs:
#             redacted_text = redacted_text[:output['start']] + output['word'] + redacted_text[output['end']:]
#         with open(output_path, 'w') as f:
#             f.write(redacted_text


if __name__ == "__main__":
    fire.Fire(main)

    # fire.Fire(main)dfdf
    


### Read XML file and extract the text


### Find the entities information in the text 