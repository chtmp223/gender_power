import json
import pandas as pd

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
    for action in actions:
        verb = lemmatizer.lemmatize(action['w'])
        if verb in verb_to_indicator and verb_to_indicator[verb] is not None:
            indicator_sum += verb_to_indicator[verb]
        else:
            pass # ignore for now
    return indicator_sum


def get_verb_to_score_dict(agency_power_frames_df, indicator_func, type):
    indicators = map(indicator_func, list(agency_power_frames_df[type]))
    return dict(zip(agency_power_frames_df['verb'], indicators))



def get_character_scores(translation):

    book_file = f'./booknlp_output/{translation}/full_text/ovid.book'
    agency_power_frames_df = pd.read_csv(f"./lexicon/FramesAgencyPower/agency_power_MODIFIED.csv")

    verb_to_power_score = get_verb_to_score_dict(agency_power_frames_df, power_indicator, "power")
    verb_to_agency_score = get_verb_to_score_dict(agency_power_frames_df, agency_indicator, "agency")

    lemmatizer = WordNetLemmatizer()

    characters = []
    genders = []
    power_scores = []
    agency_scores = []

    with open(book_file, 'r') as f:
        book_json = json.load(f)
        for char in book_json['characters']:

            if 'g' in char and char['g'] is not None:
                characters.append(char['id'])
                genders.append(char['g']['argmax'])

                # Power
                agent_power = sum_of_actions(char['agent'], verb_to_power_score, lemmatizer)
                patient_power = -1 * sum_of_actions(char['patient'], verb_to_power_score, lemmatizer)
                power = agent_power + patient_power
                power_scores.append(power)

                # Agency
                agency = sum_of_actions(char['agent'], verb_to_agency_score, lemmatizer)
                agency_scores.append(agency)
                


    char_scores = pd.DataFrame()
    char_scores['character'] = characters
    char_scores['gender'] = genders
    char_scores['power_score'] = power_scores
    char_scores['agency_score'] = agency_scores
    return char_scores

    #'characters': ['agent', 'patient', 'mod', 'poss', 'id', 'g', 'count', 'mentions']



def get_gender_sum(char_scores):
    # Get raw sums of scores based on gender
    gender_sum = char_scores.groupby(['gender']).sum()
    gender_sum.drop(columns="character", inplace = True)
    return gender_sum



def main():

    translations = ["Kline_translation", "More_translation"]

    for translation in translations:
        char_scores = get_character_scores(translation)
        char_scores.to_csv(f"./power_agency_gender/{translation}_characters_scores.csv")

        gender_sum = get_gender_sum(char_scores)
        gender_sum.to_csv(f"./power_agency_gender/{translation}_gender_sum.csv")


if __name__ == "__main__":
    main()