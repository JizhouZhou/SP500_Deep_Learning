import yfinance as yf
import pandas as pd
import pandas_datareader
import datetime

class Data():
    def __init__(self):
        pass

class YahooData(Data):
    def __init__(self, ticker, start_date, end_date, X_col):
        self.yf = YahooData.get_yf(ticker, start_date, end_date)
        self.yf[X_col + '_Return'] = self.yf[X_col].pct_change(1)
    
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
        
class FamaFrenchData(Data):
        
    def get_ff():
        temp = pandas_datareader.DataReader(name = '', data_source = 'famafrench')
        print(temp)
        
if __name__ == '__main__':
    sp500 = YahooData("^GSPC", "2010-01-01", "2020-12-31", "Close")
    ff = FamaFrenchData()
