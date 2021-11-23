from reddit_data_collector import RedditData
import pandas as pd
from datetime import datetime
from time import sleep
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


class TextDataProcessor():
    def __init__(self, query, start_date, end_date):
        temp_sub = RedditData('submission', query, start_date, end_date)
        temp_com = RedditData('comment', query, start_date, end_date)
        
        self.data = pd.merge(temp_sub.data, 
                             temp_com.data, 
                             on = 'Date',
                             suffixes = ('_sub', '_com'),)
    
    def stem(str_input):
        '''
        A stand-alone function that stems a sentence.

        Parameters
        ----------
        str_input : str
            A string variable to be stemmed.

        Raises
        ------
        ValueError
            Raise ValueError when more than 1 group in token pattern is captured.

        Returns
        -------
        words : lst
            Post stemming list.

        '''
        
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )
    
        words = token_pattern.findall(str_input)
        
        ps = PorterStemmer()
        
        words = [ps.stem(word) for word in words]
        
        return words
    
    def vectorize(self,
                  full_query = True, 
                  ana = 'word',
                  sw = 'english',
                  lc = True,
                  bi = False,
                  st = True,
                  rare_word = 10):
        '''
        Process the predictors to prepare for model training

        Parameters
        ----------
        full_query : BOOL, optional
            Include comment or not. The default is True.
        ana : str, optional
            Count Vectorizer parameter analyzer. The default is 'word'.
        sw : str, optional
            Count Vectorizer parameter stop_words. The default is 'english'.
        lc : BOOL, optional
            Count Vectorizer parameter lowercase. The default is True.
        bi : BOOL, optional
            Count Vectorizer parameter binary. The default is False.
        st: BOOL, optional
            Count Vectorizer parameter stemmer. The default is False.
        rare_word : int, optional
            Set the threshold to remove typo. The default is 10.

        Returns
        -------
        None.

        '''
        if st:
            Vec = CountVectorizer(analyzer = ana,
                                  stop_words = sw,
                                  lowercase = lc,
                                  binary = bi,
                                  tokenizer = TextDataProcessor.stem)
        else:
            Vec = CountVectorizer(analyzer = ana,
                                  stop_words = sw,
                                  lowercase = lc,
                                  binary = bi)
            
        if full_query:
            temp_d = [' '.join(x) for x in self.data.Title + ' ' + self.data.Text_sub + ' ' + self.data.Text_com]
        else:
            temp_d = [' '.join(x) for x in self.data.Title + ' ' + self.data.Text_sub]
        
        data_X_t = Vec.fit_transform(temp_d)
        
        features = Vec.get_feature_names()
        
        temp = pd.DataFrame(data_X_t.toarray(), columns = features)

        temp = temp.loc[:, temp.sum(axis = 0) > rare_word]
        
        self.vec_data = temp
        
        print(self.data.head)
        
        
        
if __name__ == '__main__':
    reddit_10 = TextDataProcessor('finance',
                                  datetime(2019,3,1), 
                                  datetime(2019,3,2))
    reddit_10.vectorize()
    
    