# [CS690F F22] Gender and Power Dynamics in Latin Narratives



## Running Power/Agency statistics:
#### Full text statistics:
From the top level folder run: `python ./power_agency_gender/power_score_gender.py`

#### Per character/selected stories:
Code available in `booknlp_output/gender_agent_booknlp.ipynb`

Note that the Tereus/Philomela output is manually edited to be more accurate. The code can be run on other stories, but power/agency scores will be less accurate per character.

## Running Topic Modeling code: 
- `topic-modeling/lda-train.py`: training file for LDA topic model. From the top-level folder, run `python ./topic-modeling/lda-train.py ovid custom`. This will run topic modeling on our dataset with a custom stopword list. 
- `topic-modeling/lda-prep.ipynb`: segment our dataset into documents for training. 
- `topic-modeling/lda-analysis.ipynb`: analyses on LDA results

## Running Word Cloud :
- `booknlp_output/entire_wordcloud.ipynb`: analysis of word clouds on the entire text
