import re
from config import QA_PROMPT

def clean_text(text):
    text = str(text).strip()
    return text

def prepare_document(question, answer):
    return QA_PROMPT.format(question=question, answer=answer)
