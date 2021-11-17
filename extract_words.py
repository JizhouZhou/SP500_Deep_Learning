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

# I couldn't get nltk to import, so I manually defined the stopword list
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
        









#---------------------------
# CHUNK AND CLEAN EACH NOVEL
#---------------------------

# function for chunking and cleaning each novel
def chunk_and_clean(f, special_words):
    
    # open the textfile passed to the function
    with open(f) as txtfile:
        
        # read each line
        lines = txtfile.readlines()
    
    # strip ending newlines from each string of words
    rstrip_nl = [line.rstrip('\n') if line != '\n' else line for line in lines]
    
    # join the remaining strings by the blank spaces from the stripping
    join_by_blank_lines = ' '.join(rstrip_nl)
    
    # split at the newlines separating text into clumps of lines in Project G
    split_nl = join_by_blank_lines.split('\n')
    
    # initialize a list for storing all of the chunked text
    all_chunks = []
    
    # only keep the chunks that have at least more than just whitespace
    for chunk in split_nl:
        if chunk != '' and chunk != ' ':
            all_chunks.append(chunk)
    
    # delete all the chunked text not related to the actual novel
    book_chunks = all_chunks[9:(len(all_chunks)-51)]
    
    # remove leading and trailing whitespace from each chunk
    rm_ws = [chunk.lstrip().rstrip() for chunk in book_chunks]
    
    # initialize a list for storing partially cleaned chunks
    semi_clean_chunks = []
    
    # for each chunk
    for chunk in rm_ws:
        
        # break the text chunk into words
        words = chunk.split()
        
        # initialize a list for storing words with the dashes taken out
        no_dash = []
        
        # for each word
        for word in words:
            
            # split the word up by the short dash(es) (if it has any)
            split_short_dash = [word for word in word.split('-')]
            
            # split the leftovers by the long dash(es) (if there are any)
            split_long_dash = [word.split('—') for word in split_short_dash]
            
            # initialize a list to combine each split list
            combined = []
            
            # for each list of words from the splits
            for word_list in split_long_dash:
                
                # add these to the combined list
                combined += word_list
                
            # add the combined list to the list of words with no dashes    
            no_dash += combined
            
        # reassign the words to be the former words, except with no dashes
        words = no_dash
        
        # remove any non-alphabetical characters from each word
        rm_nab = [re.compile('[^a-zA-Z]').sub('', word) for word in words]
        
        # initialize a list for the words to keep for the cleaned chunk
        words_to_keep = []
        
        # for each word with the non-alphabetical characters removed
        for word in rm_nab:
            
            # convert the word to lowercase
            word = word.lower()
            
            # if word is not a stopword and not in title of book
            if (not(word in stopwords)) and (not(word in special_words)):
                
                # keep the word
                words_to_keep.append(word)
                
        # if the chunk contains at least 5 or more words
        if len(words_to_keep) >= 5:
            
            # append to semi-clean chunk list by joining words by whitespace
            semi_clean_chunks.append(' '.join(words_to_keep))
    
    # remove the leading and trailing whitespace again
    rm_ws_again = [chunk.lstrip().rstrip() for chunk in semi_clean_chunks]
    
    # initialize a list to store the cleaned chunks
    clean_chunks = []
    
    # remove empty chunks
    for chunk in rm_ws_again:   
        if chunk != '':
            clean_chunks.append(chunk)
            
    # return the list of cleaned chunks
    return clean_chunks

# call function for each novel and pass title words to remove
aiw_initial = chunk_and_clean('11-0.txt', ['alice', 'wonderland'])
md_initial  = chunk_and_clean('15-0.txt', ['moby', 'dick', 'whale', 'ishmael'])
pp_initial  = chunk_and_clean('16-0.txt', ['peter', 'pan', 'wendy'])

# remove chunks pertaining to the chapter lists (and etymology for moby-dick)
aiw_final = aiw_initial[1:]
md_final  = md_initial[96:]
pp_final  = pp_initial[2:]


#--------------------------------------------
# DETERMINE THE TOP 10000 MOST FREQUENT WORDS
#--------------------------------------------

# combine the chunks from all novels
all_final = aiw_final + md_final + pp_final

# make lists of of the list of words in each chunk
all_chunk_lists = [chunk.split() for chunk in all_final]

# initialize a list to store all words
all_combined = []

# for each list of words of the chunks
for chunk_list in all_chunk_lists:
    
    # add the list of words into one list
    all_combined += chunk_list

# use Counter function to find the top 10000 most frequently occurring words
most_freq = Counter(all_combined).most_common(10000)

# form a dictionary of the words and how many times they occur
most_freq_dict = dict(most_freq)

# store the most frequent words by using the keys of the dictionary
most_freq_words = most_freq_dict.keys()


#--------------------------------------------
# REMOVE WORDS NOT IN TOP 10000 MOST FREQUENT
#--------------------------------------------

# function for removing the words from each chunk that are not in top 10000 
def remove_uncommon_words(novel_chunks):
    
    # initialize a list to store the chunks with only the common words listed
    common_word_chunks = []
    
    # for each chunk in the novel passed
    for chunk in novel_chunks:
        
        # split the chunk into individual words
        words = chunk.split()
        
        # initialize a list for storing the words to keep
        words_to_keep = []
        
        # for each word
        for word in words:
            
            # if the word is in the top 10000
            if word in most_freq_words:
                
                # keep the word
                words_to_keep.append(word)
                
        # if the new chunk is still 5 or more words long
        if len(words_to_keep) >= 5:
            
            # append to common word chunk list by joining words by whitespace
            common_word_chunks.append(' '.join(words_to_keep))
    
    # remove the leading and trailing whitespace
    rm_ws = [chunk.lstrip().rstrip() for chunk in common_word_chunks]
    
    # initialize a list to store the cleaned chunks
    clean_chunks = []
    
    # remove empty chunks
    for chunk in rm_ws:   
        if chunk != '':
            clean_chunks.append(chunk)
    
    # return the cleaned chunks
    return clean_chunks

# call function for each novel
aiw_final_common = remove_uncommon_words(aiw_final)
md_final_common  = remove_uncommon_words(md_final)
pp_final_common  = remove_uncommon_words(pp_final)


#-----------------------------
# CREATE THE PROCESSED DATASET
#-----------------------------

# calculate the number of chunks for each novel
num_0 = len(aiw_final_common)
num_1 = len(md_final_common)
num_2 = len(pp_final_common)

# create a list for the labels that will be replicated
labels = np.array([0, 1, 2])

# rep each label based on the number of chunks in the corresponding novel
labels_array = np.repeat(labels, [num_0, num_1, num_2]).astype('float32')

# create the corpus by combining the final chunks from all novels
corpus = aiw_final_common + md_final_common + pp_final_common

# form a dataframe of the chunk and corresponding label
processed_data = pd.DataFrame({'Chunk': corpus, 'Label': labels_array})

# write the dataframe to a csv file
processed_data.to_csv('processed_data.csv')

