#! /bin/bash
cd ~/mallet


# Setting inputs/global variables ---- 
BNAME=$1                                  

CORPUS=${BNAME/-?*/}                     
MALLET_DIR=../gender_power/topic-modeling/output/              

INPUT=../gender_power/topic-modeling/input/
SEQ=$INPUT/$BNAME.seq                  
TSV=$INPUT/$BNAME.tsv

# Corpus status ----
if [ $CORPUS = king ]
then 
  TOPICS='20 30'
elif [ $CORPUS = ovid ]; 
then
  TOPICS='20, 30, 50, 100, 200'
else
  echo "Can't find corpus $CORPUS"
fi 

echo "Importing data"
bin/mallet import-dir --input $INPUT --output $SEQ