# Script to train LDA topic model 
import little_mallet_wrapper as lmw
import pandas as pd
import sys
from collections import Counter
from nltk.corpus import stopwords

# Setting global variables ----
path_to_mallet = '~/mallet/bin/mallet'
path_to_proj = '../gender_power/'
INPUT = "./topic-modeling/input/"


def stopwords_list(data, nltk_stopwords): 
  '''
  Words that occur in more than 50% of the documents
  '''
  stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 'amp', 'could', 'may', 'might', 'must', 'would', 'should', 'even', 'though', 'one']
  if nltk_stopwords == "nltk":
    for item in stopwords.words('english'):
      stop.append(item)
  else: 
    all_words = []
    for row in data: 
      unq = set(row.split())
      all_words.append(unq)
    all_count = Counter([i for item in all_words for i in item])
    for k,v in all_count.items(): 
      if v > len(data)/2: 
        stop.append(k)
  return stop


def data(title_to_train, stop_corpus):
  '''
  Preparing data for training
  '''
  train_df = pd.read_csv(INPUT + title_to_train+ ".tsv", sep='\t')
  stopwords = stopwords_list(train_df['text'].tolist(), stop_corpus)
  training_data = [lmw.process_string(t, lowercase=True, remove_stop_words=True, stop_words=stopwords) for t in train_df['text'].tolist()]
  training_data = [d for d in training_data if d.strip()] 
  return training_data


def training(training_data, title_to_train, num_topics): 
  '''
  Train LDA model with given number of topics
  '''
  
  # Setting output path ----
  output_directory_path = path_to_proj + 'topic-modeling/output/' + title_to_train
  path_to_training_data           = output_directory_path + '/training.txt'
  path_to_formatted_training_data = output_directory_path + '/mallet.training'
  path_to_model                   = output_directory_path + '/mallet.model.' + str(num_topics)
  path_to_topic_keys              = output_directory_path + '/mallet.topic_keys.' + str(num_topics)
  path_to_topic_distributions     = output_directory_path + '/mallet.topic_distributions.' + str(num_topics)
  path_to_word_weights            = output_directory_path + '/mallet.word_weights.' + str(num_topics)
  path_to_diagnostics             = output_directory_path + '/mallet.diagnostics.' + str(num_topics) + '.xml'

  lmw.import_data(path_to_mallet,
                path_to_training_data,
                path_to_formatted_training_data,
                training_data)
  lmw.print_dataset_stats(training_data)
  lmw.train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      path_to_model,
                      path_to_topic_keys,
                      path_to_topic_distributions,
                      path_to_word_weights,
                      path_to_diagnostics,
                      num_topics)


def main(): 
  title_to_train = sys.argv[1]
  stop_corpus = sys.argv[2]                       # 'nltk' or 'custom'
  num_topics = [5,10,15,20,30,50,100,200]         # Comment out 10 if run again 
  training_data = data(title_to_train, stop_corpus)
  for num in num_topics: 
    print("Training model with " + str(num) + " topics")
    training(training_data, title_to_train, num)


if __name__ == "__main__":
  main()