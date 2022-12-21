import json
import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

  

def power_indicator(power_string):
    if power_string == "power_agent":
        return 1
    elif power_string == "power_equal":
        return 0
    elif power_string == "power_theme":
        return -1

def agency_indicator(agency_string):
    if agency_string == "agency_pos":
        return 1
    elif agency_string == "agency_equal":
        return 0
    elif agency_string == "agency_neg":
        return -1


def sum_of_actions(actions, verb_to_indicator, lemmatizer):
    indicator_sum = 0
    count = 0
    for action in actions:
        verb = lemmatizer.lemmatize(action['w'])
        if verb in verb_to_indicator and verb_to_indicator[verb] is not None:
            indicator_sum += verb_to_indicator[verb]
            count += 1
        else:
            pass # ignore for now
    return indicator_sum, count


def mean_of_actions(actions, verb_to_indicator, lemmatizer):
    if len(actions) == 0:
        return 0
    indicator_sum = 0
    for action in actions:
        verb = lemmatizer.lemmatize(action['w'])
        if verb in verb_to_indicator and verb_to_indicator[verb] is not None:
            indicator_sum += verb_to_indicator[verb]
        else:
            pass # ignore for now
    return indicator_sum / len(actions)


def get_verb_to_score_dict(agency_power_frames_df, indicator_func, type):
    indicators = map(indicator_func, list(agency_power_frames_df[type]))
    return dict(zip(agency_power_frames_df['verb'], indicators))

def get_dom_score(char, word_type, dom_to_score):
    score = 0
    for mod in char[word_type]:
        mod_w = mod['w']
        if mod_w in dom_to_score:
            score += dom_to_score[mod_w]
    return score

def get_character_scores(translation):

    book_file = f'./booknlp_output/{translation}/full_text/ovid.book'
    agency_power_frames_df = pd.read_csv(f"./lexicon/FramesAgencyPower/agency_power_MODIFIED.csv")

    verb_to_power_score = get_verb_to_score_dict(agency_power_frames_df, power_indicator, "power")
    verb_to_agency_score = get_verb_to_score_dict(agency_power_frames_df, agency_indicator, "agency")

    lemmatizer = WordNetLemmatizer()

    dominance_df = pd.read_csv(f"./lexicon/NRC-VAD-Lexicon/BipolarScale/OneFilePerDimension/dominance-NRC-VAD-Lexicon.txt", sep='\t', names=['word', 'value'])
    dom_to_score = dict(zip(dominance_df['word'], dominance_df['value']))

    characters = []
    genders = []
    power_scores = []
    agency_scores = []
    dom_scores = []

    with open(book_file, 'r') as f:
        book_json = json.load(f)
        for char in book_json['characters']:

            if 'g' in char and char['g'] is not None:
                characters.append(char['id'])
                genders.append(char['g']['argmax'])

                # Power
                agent_power, agent_count = sum_of_actions(char['agent'], verb_to_power_score, lemmatizer)
                patient_power, patient_count = sum_of_actions(char['patient'], verb_to_power_score, lemmatizer)
                power_count = agent_count + patient_count
                if power_count > 0:
                    power_scores.append(agent_power - patient_power)
                else:
                    power_scores.append(0)

                # Agency
                agency, agency_count = sum_of_actions(char['agent'], verb_to_agency_score, lemmatizer)
                if agency_count > 0:
                    agency_scores.append(agency)
                else:
                    agency_scores.append(0)

                # Dominance
                mod_score = get_dom_score(char, 'mod', dom_to_score)
                poss_score = get_dom_score(char, 'poss', dom_to_score)
                dom_scores.append(mod_score + poss_score)
                

    char_scores = pd.DataFrame()
    char_scores['character'] = characters
    char_scores['gender'] = genders
    char_scores['power_score'] = power_scores
    char_scores['agency_score'] = agency_scores
    char_scores['dom_score'] = dom_scores
    return char_scores

    #'characters': ['agent', 'patient', 'mod', 'poss', 'id', 'g', 'count', 'mentions']



def get_gender_statistics(char_scores):
    gender_sum = char_scores.groupby('gender').agg(power_sum=('power_score', 'sum'), agency_sum=('agency_score', 'sum'), dominance_sum=('dom_score', 'sum'), count=('character', 'nunique'), power_mean=('power_score', 'mean'), agency_mean=('agency_score', 'mean'), dom_mean=('dom_score', 'mean'))
    return gender_sum

def get_top_power_agency(char_scores):
    top_power = char_scores.sort_values(by=['power_score'], ascending = False)
    top_power = top_power.head(10)
    top_agency = char_scores.sort_values(by=['agency_score'], ascending = False)
    top_agency = top_agency.head(10)
    return top_power, top_agency

def main():

    translations = ["Kline_translation", "More_translation"]

    for translation in translations:
        char_scores = get_character_scores(translation)
        char_scores.to_csv(f"./power_agency_gender/{translation}/characters_scores.csv")

        gender_stats = get_gender_statistics(char_scores)
        gender_stats.to_csv(f"./power_agency_gender/{translation}/gender_stats.csv")

        top_power, top_agency = get_top_power_agency(char_scores)
        top_power.to_csv(f"./power_agency_gender/{translation}/top_power.csv")
        top_agency.to_csv(f"./power_agency_gender/{translation}/top_agency.csv")
        print(f"Successfully saved {translation} .csv files")

if __name__ == "__main__":
    main()