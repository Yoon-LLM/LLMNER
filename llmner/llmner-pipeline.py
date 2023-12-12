# https://vkhangpham.medium.com/build-a-custom-ner-pipeline-with-hugging-face-a84d09e03d88

def postprocess_llmner(text, tag):
    """
    Postprocess the output of LLM-NER.
    ---
    >>> postprocess_llmner("hello, I'm John Smith", "redacted")
    [{'entity': 'REDACTED', 'index': 6, 'word': 'John', 'start': 11, 'end': 15}]
    """
    NotImplementedError


def llmner_pipeline(text):
    """
    Named Entity Recognition
    param: text: str
    return: dict
    ---
    >>> ner("hello, I'm John Smith")
    [{'entity': 'I-PER', 'score': 0.997681, 'index': 6, 'word': 'John', 'start': 11, 'end': 15}, {'entity': 'I-PER', 'score': 0.99473864, 'index': 7, 'word': 's', 'start': 16, 'end': 17}, {'entity': 'I-PER', 'score': 0.9801539, 'index': 8, 'word': '##mith', 'start': 17, 'end': 21}]
    """
    NotImplementedError



