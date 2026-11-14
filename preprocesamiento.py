
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def spacy_preprocessor(text):
    doc = nlp(text)

    tokens_limpios = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens_limpios.append(token.lemma_.lower())

    return " ".join(tokens_limpios)
