"""
NER for sermons in content.dat - for any NE class
"""
import os
import pandas as pd
import numpy as np
import argparse

import dacy
nlp = dacy.load("large")

import nltk.data
from nltk.stem.snowball import DanishStemmer
stemmer = DanishStemmer()

if __name__ == "__main__":

    print("\n--------\n")

    """
    Get input arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--entity_class', type=str, required=False,
                        default="LOC",
                        help='DaCy Named Entitiy Tag to extract. Defaults to LOC')

    args = parser.parse_args()
    entity_class = args.entity_class

    """
    First processing to create NER outputs
    """
    df = pd.read_csv(os.path.join("data", "sermon_content.dat"), 
                        encoding='utf-8', 
                        header = 0, 
                        index_col = None)

    content = df["content"].tolist()
    fnames = df["id"].tolist()
    tokenizer = nltk.data.load(os.path.join("tokenizers", "punkt", "norwegian.pickle"))

    entity_list = []
    i = 0
    for i, text in enumerate(content):
    #for i, text in enumerate(content[:50]):
        print(f"file {i}")
        # sentence disambiguation
        sentences = tokenizer.tokenize(text)
        text_entities = []
        # extract entities from sentences
        for sent in sentences:
            text_entities.append([(ent.text, ent.label_) for ent in nlp(sent).ents])
            #if textblob.entities:
            #    text_entities.append(textblob.entities)
        entity_list.append([fnames[i],text_entities])
        
    df_ner = pd.DataFrame(entity_list)
    df_ner.columns = ["id", "NE"]
    df_ner.to_csv(os.path.join("data", "content_entities_dacy.dat"), index = False)


    """
    Extract all occurrences of the entitiy of interest at sentence level for each document.
    """
    entities = df_ner["NE"].tolist()
    fname = df_ner["id"]

    out = []
    for i, doc in enumerate(entities):
        for ii, sent in enumerate(doc):
            if sent:
                for entity in sent:
                    if entity[1] == entity_class:
                        out.append([fname[i], ii, entity[0]])
    entitiy_df = pd.DataFrame(out)
    entitiy_df.columns = ["fname","sentence", entity_class]

    # Clean up any punctuation, whitespace, etc
    entitiy_df[entity_class] = entitiy_df[entity_class].str.replace('[^\w\s]','', regex=True)
    # To lower
    entitiy_df[entity_class] = entitiy_df[entity_class].str.lower()
    # Replace any empty cells with Nan; remove NaN
    entitiy_df[entity_class].replace('', np.nan, inplace=True)
    entitiy_df = entitiy_df.dropna()

    # Stem remaining names with Snowball
    # This is far from perfect but it gets rid of things like possessives
    entitiy_df[entity_class] = entitiy_df[entity_class].apply(lambda x: stemmer.stem(x))
    entitiy_df.to_csv(os.path.join("data",f"content_{entity_class}_dacy.dat"), index = False)

    # """
    # Join with metadata
    # """
    # meta = pd.read_excel(os.path.join("data", "meta", "Joined_Meta.xlsx"))
    # ner_people_with_meta = entitiy_df.merge(meta, 
    #                                     left_on='fname', 
    #                                     right_on='ID-dok', 
    #                                     how='left')
    # ner_people_with_meta.to_csv(os.path.join("data", "ner_people_with_meta.dat", 
    #                                     encoding='utf8'))
