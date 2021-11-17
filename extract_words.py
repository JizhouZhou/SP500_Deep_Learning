"""
Chevy Robertson (crr78@georgetown.edu)
ANLY 590: Neural Networks & Deep Learning
Group Semester Project: Word Extraction
11/17/2021
"""


#-------- 
# IMPORTS
#--------

import re
import numpy as np
import pandas as pd
from collections import Counter


#----------
# STOPWORDS
#----------

# I couldn't get nltk's stopwords to work, so I manually defined the stopword list
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
             'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
             'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
             'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
             'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
             'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
             'very', 'can', 'will', 'just', 'don', 'should', 'now', 'b', 'c',
             'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# read in data
sub_temp = pd.read_csv('sub_temp.csv')

# delete unnamed column
sub_temp = sub_temp.drop('Unnamed: 0', axis=1)

# cleans data to extract all words from different times assoc. w/ same date
def extract_words(df, stopwords):
    
    # initialize a list for storing the combined text
    combined_text = []
    
    # loop through the index of the dataframe
    for i in range(0, len(df)):

        # grab the text and convert from float to string
        st_ex = str(df['selftext'][i])
        
        # split by whitespace
        st_ex_split = st_ex.split()
        
        # remove any non-alphabetical characters from each word
        rm_nab = [re.compile('[^a-zA-Z\d]').sub('', word) for word in st_ex_split]
        
        # initialize a list to store the words that should be kept
        words_to_keep = []
                
        # for each word with the non-alphabetical characters removed
        for word in rm_nab:
                    
            # convert the word to lowercase
            word = word.lower()
            
            # if word is not a stopword
            if (not(word in stopwords)) and word != '':
                
                # keep the word
                words_to_keep.append(word)
                
        # add the words to keep to the combined text
        combined_text += words_to_keep
        
    # return the combined text
    return combined_text
        
# call the function to get all the words and save this to a variable
all_words = extract_words(sub_temp, stopwords)
        
