import re

QA_PROMPT = """Question: {question}\n\nAnswer: {answer}"""

def clean_text(text):
    text = str(text).strip().lower()
    return text

def prepare_document(question, answer):
    return QA_PROMPT.format(question=question, answer=answer)
