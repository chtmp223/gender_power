# Script to train LDA topic model 
import little_mallet_wrapper as lmw
import pandas as pd
import sys


# Setting global variables ----
path_to_mallet = '~/mallet/bin/mallet'
path_to_proj = '../gender_power/'
INPUT = "./topic-modeling/input/"


def training(title_to_train, num_topics): 
  '''
  Train LDA model with given number of topics
  '''
  # Preparing training data ---
  train_df = pd.read_csv(INPUT + title_to_train+ ".tsv", sep='\t')
  training_data = [lmw.process_string(t) for t in train_df['text'].tolist()]
  training_data = [d for d in training_data if d.strip()]

  # Setting output path ----
  output_directory_path = path_to_proj + 'topic-modeling/output'
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
  num_topics = [5,10,15,20,30,50,100,200]
  for num in num_topics: 
    print("Training model with " + str(num) + " topics")
    training(title_to_train, num)
  print("Finished!")


if __name__ == "__main__":
  main()