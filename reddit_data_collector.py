import requests
import pandas as pd
import datetime
from time import sleep

class Data():
    def __init__(self):
        '''
        Super class placeholder.

        Returns
        -------
        None.

        '''
        pass
    
class RedditData(Data):
    def __init__(self, dtype, query, start_date, end_date, freq = 1):
        delta = datetime.timedelta(days = freq)
        if dtype == 'submission':
            self.data = pd.DataFrame(columns = ['Date', 'Title', 'Text'])
            while start_date <= end_date:
                print(start_date.strftime(('%Y-%m-%d')))
                try:
                    temp = self.clean_reddit(dtype,
                                             query,
                                             start_date.strftime('%s'), 
                                             (start_date+delta).strftime('%s'))
                    self.data = self.data.append({'Date' : start_date, 'Title': temp['title'].values, 'Text': temp['selftext'].values}, 
                                                 ignore_index = True)
                except:
                    pass
                
                start_date += delta
        elif dtype == 'comment':
            self.data = pd.DataFrame(columns = ['Date', 'Text'])
            while start_date <= end_date:
                print(start_date.strftime(('%Y-%m-%d')))
                try:
                    temp = self.clean_reddit(dtype,
                                             query,
                                             start_date.strftime('%s'), 
                                             (start_date+delta).strftime('%s'))
                    self.data = self.data.append({'Date': start_date, 'Text': temp['body'].values},
                                                 ignore_index = True)
                except:
                    pass
                
                start_date += delta
            
    
    def get_pushshift_data(data_type, **kwargs):
        '''
        A class method to access data from Reddit API

        Parameters
        ----------
        data_type : str
            Data to access 'comment' or 'submission'.
        **kwargs : keyword arguments
            Parameters.

        Returns
        -------
        json
            Raw Data.

        '''
        base_url = f"https://api.pushshift.io/reddit/search/{data_type}/"
        payload = kwargs
        request = requests.get(base_url, params=payload)
        
        return request.json()
    
    def get_reddit(dtype, query, start_date, end_date, size = 500, sort_type = 'score', sort = 'desc'):
        '''
        A class method to call get_pushshift_data organize in pandas DataFrame.

        Parameters
        ----------
        dtype : str
            Data to access 'comment' or 'submission'.
        query : str
            Key word to search from Reddit to access.
        start_date : datatime
            First date to be accessed format "YYYY-MM-DD".
        end_date : datetime
            Last date to be accessed format "YYYY-MM-DD".
        size : int, optional
            Max number of Reddits. The default is 500.
        sort_type : str, optional
            Sort by. The default is 'score'.
        sort : str, optional
            Ascending or descending. The default is 'desc'.

        Returns
        -------
        DataFrame
            DataFrame of raw data.

        '''
        data = RedditData.get_pushshift_data(data_type = dtype,
                                             q = query,
                                             size = size,
                                             sort_type = 'score',
                                             after = start_date,
                                             before = end_date,
                                             sort = sort).get("data")
        return pd.DataFrame(data)
    
    def clean_reddit(self, dtype, query, start_date, end_date, size = 500, sort_type = 'score', sort = 'desc'):
        '''
        A class method to clean raw data.

        Parameters
        ----------
        dtype : str
            Data to access 'comment' or 'submission'.
        query : str
            Key word to search from Reddit to access.
        start_date : datatime
            First date to be accessed format "YYYY-MM-DD".
        end_date : datetime
            Last date to be accessed format "YYYY-MM-DD".
        size : int, optional
            Max number of Reddits. The default is 500.
        sort_type : str, optional
            Sort by. The default is 'score'.
        sort : str, optional
            Ascending or descending. The default is 'desc'.

        Returns
        -------
        temp : DataFrame
            DataFrame with only necessary columns.

        '''
        temp = RedditData.get_reddit(dtype, query, start_date, end_date)
        if dtype == 'submission':
            temp = temp[['created_utc', 'title', 'selftext']]
        
        if dtype == 'comment':
            temp = temp[['created_utc', 'body']]
        
        temp['created_utc'] = [datetime.datetime.fromtimestamp(time) for time in temp['created_utc']]
        
        return temp


# if __name__ == '__main__':
    # reddit_20120101_S = RedditData('submission',
    #                                 'finance',
    #                                 datetime.datetime(2017,1,1), 
    #                                 datetime.datetime(2017,1,10))
    
    # reddit_20120101_C = RedditData('comment',
    #                                 'finance',
    #                                 datetime.datetime(2017,1,1), 
    #                                 datetime.datetime(2017,1,10))
    
    # data_s = RedditData.clean_reddit('submission',
    #                                  'finance',
    #                                  datetime.datetime(2017,1,1).strftime('%s'), 
    #                                  datetime.datetime(2017,1,2).strftime('%s'))
    
    # data_c = reddit_20120101.clean_reddit('comment',
    #                                       'finance',
    #                                       datetime.datetime(2017,1,1).strftime('%s'), 
    #                                       datetime.datetime(2017,1,2).strftime('%s'))