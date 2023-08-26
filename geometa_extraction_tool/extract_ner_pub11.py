from pprint import pprint
from PIL import Image
import piexif
import os
from tqdm import tqdm
import pickle as pkl
import nltk
# import spacy
import pandas as pd
import locationtagger
import spacy
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


nlp = spacy.load('en_core_web_sm')
codec = 'ISO-8859-1'  # or latin-1
# nltk.downloader.download('maxent_ne_chunker')
# nltk.downloader.download('words')
# nltk.downloader.download('treebank')
# nltk.downloader.download('maxent_treebank_pos_tagger')
# nltk.downloader.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def cap_to_location(text):
    text_change = text

    # extracting entities.
    # place_entity = locationtagger.find_locations(text=text)
    # countries = place_entity.countries
    # regions = place_entity.regions
    # cities = place_entity.cities

    doc = nlp(text_change)
    entity_list = []
    for entity in doc.ents:
        if entity.label_ == 'GPE':
            entity_text = entity.text
            entity_list.append(entity_text)

    # entity_list = []
    # nes = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    # for ne in nes:
    #     if type(ne) is nltk.tree.Tree:
    #         if (ne.label() == 'GPE'):
    #             l = []
    #             for i in ne.leaves():
    #                 l.append(i[0])
    #             s = u' '.join(l)
    #             if not (s in entity_list):
    #                 entity_list.append(s)

    # st = StanfordNERTagger(
    #     '/home/zilun/RS5M_v4/nips_rebuttal/geometa/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
    #     '/home/zilun/RS5M_v4/nips_rebuttal/geometa/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner-4.2.0.jar',
    #     encoding='utf-8'
    # )
    # tokenized_text = word_tokenize(text_change)
    # classified_text = st.tag(tokenized_text)
    # entity_list = []
    # for entity in classified_text:
    #     if entity[1] == 'LOCATION':
    #         entity_text = entity[0]
    #         entity_list.append(entity_text)

    # # use caseless
    # st = StanfordNERTagger(
    #     "/home/zilun/RS5M_v4/nips_rebuttal/geometa/stanford-ner-3.5.2/stanford-corenlp-caseless-2015-04-20-models/edu/stanford/nlp/models/ner/english.all.3class.caseless.distsim.crf.ser.gz",
    #     "/home/zilun/RS5M_v4/nips_rebuttal/geometa/stanford-ner-3.5.2/stanford-ner-2015-04-20/stanford-ner-3.5.2.jar",
    #     encoding="utf-8"
    # )
    # tokenized_text = word_tokenize(text_change)
    # classified_text = st.tag(tokenized_text)
    # entity_list = []
    # for entity in classified_text:
    #     if entity[1] == 'LOCATION':
    #         entity_text = entity[0]
    #         entity_list.append(entity_text)

    return entity_list


def main_keyword_extract():
    pub11_pkl_path = "/media/zilun/wd-16/RS5M_v3/pub11/RS5M_pub11_label.pkl"
    pub11_df = pkl.load(open(pub11_pkl_path, "rb"))
    img_names = pub11_df["img_name"].tolist()
    caps = pub11_df["text"].tolist()
    assert len(caps) == len(img_names)
    entity_list = []
    geo_location_count = 0
    for i in tqdm(range(len(img_names))):
        img_name = img_names[i]
        cap = caps[i]
        entitys = cap_to_location(cap)
        if len(entitys) > 0:
            geo_location_count += 1
        entity_list.append(entitys)

    geometa_df = pd.DataFrame({
        "img_name": img_names,
        "text": caps,
        "entity": entity_list,
    })
    print(geo_location_count)
    geometa_df.to_csv("RS5M_pub11_geolocation.csv", index=False)

    return geometa_df


def main():
    main_keyword_extract()


if __name__ == '__main__':
    main()
