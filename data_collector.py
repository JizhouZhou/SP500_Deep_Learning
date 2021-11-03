import yfinance as yf

class Data():
    def __init__(self):
        pass
    
    def get_yf(self, ticker, start_date, end_date):
        '''
        A class method to access data from yahoo finance.

        Parameters
        ----------
        ticker : str
            Ticker(s) to  be accessed seperated by space.
        start_date : str
            First date to be accessed format "YYYY-MM-DD"
        end_date : str
            Last date to be accessed format "YYYY-MM-DD"

        Returns
        -------
        DataFrame
            Historical market data from Yahoo! finance.

        '''
        return yf.download(ticker, start = start_date, end = end_date)


if __name__ == '__main__':
    try1 = Data()
    try1.sp500 = try1.get_yf("^GSPC", "2010-01-01", "2020-12-31")
