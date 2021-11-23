import yfinance as yf
import pandas as pd
import pandas_datareader
from datetime import datetime

class Data():
    def __init__(self):
        '''
        Super class placeholder.

        Returns
        -------
        None.

        '''
        pass

class YahooData(Data):
    def __init__(self, ticker, start_date, end_date, X_col):
        '''
        A class method to generate an instance of YahooData.

        Parameters
        ----------
        ticker : str
            Ticker(s) to  be accessed seperated by space.
        start_date : str
            First date to be accessed format "YYYY-MM-DD".
        end_date : str
            Last date to be accessed format "YYYY-MM-DD".
        save_path : str, optional
            Path to save file. The default is ''.

        Returns
        -------
        None.

        '''
        self.yf = YahooData.get_yf(ticker, start_date, end_date)
        self.yf[X_col + '_Return'] = self.yf[X_col].pct_change(1)
        self.yf[X_col + '_Sign'] = (self.yf[X_col].pct_change(1)) >= 0
    
    def get_yf(ticker, start_date, end_date, save_path = ''):
        '''
        A class method to access data from yahoo finance.

        Parameters
        ----------
        ticker : str
            Ticker(s) to  be accessed seperated by space.
        start_date : str
            First date to be accessed format "YYYY-MM-DD".
        end_date : str
            Last date to be accessed format "YYYY-MM-DD".
        save_path : str, optional
            Path to save file. The default is ''.

        Returns
        -------
        DataFrame
            Historical market data from Yahoo! finance.

        '''
        
        temp = yf.download(ticker, start = start_date, end = end_date)
        
        if save_path != '':
            temp.to_csv(save_path)
        
        return temp
    
    def get_return(col):
        '''
        A class method to calculate daily return.

        Parameters
        ----------
        col : DataFrame 
            Column to be converted.

        Returns
        -------
        DataFrame
            Price converted return data.

        '''
        
        return col.pct_change(1)
    
    def get_sign(col):
        '''
        A class method to calculate daily return.

        Parameters
        ----------
        col : DataFrame 
            Column to be converted.

        Returns
        -------
        DataFrame
            Price converted return data.

        '''
        
        return (col.pct_change(1))**0
        
class FamaFrenchData(Data):
    def __init__(self, start_date, end_date):
        '''
        A class method to generate an instance of FamaFrenchData.

        Parameters
        ----------
        start_date : datetime
            First date to be accessed.
        end_date : datetime
            First date to be accessed.

        Returns
        -------
        None.

        '''
        self.ff = FamaFrenchData.get_ff(start_date, end_date)
        
    def get_ff(start_date, end_date):
        '''
        A class method to access data from yahoo Fama French Data Library.

        Parameters
        ----------
        start_date : datetime
            First date to be accessed.
        end_date : datetime
            First date to be accessed.

        Returns
        -------
        None.

        '''
        temp = pandas_datareader.DataReader(name = 'F-F_Research_Data_Factors_daily',
                                            data_source = 'famafrench',
                                            start = start_date,
                                            end = end_date)
        return(temp[0])
        
if __name__ == '__main__':
    sp500 = YahooData("^GSPC", "2010-01-01", "2020-12-31", "Close")
    ff = FamaFrenchData(datetime(2010, 1, 1), datetime(2020, 12, 31))
    
    