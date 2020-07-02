import requests
import pandas
import io
import os
import yfinance
import threading

def getDataframeFromUrl(url):
    dataString = requests.get(url).content
    parsedResult = pandas.read_csv(io.StringIO(dataString.decode('utf-8')))
    return parsedResult

# def fetchDailyStockPrice(ticker, folder):
#     url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey=IP95X3TC4M4YIPJ2&datatype=csv'.format(ticker = ticker)
#     dailyData = getDataframeFromUrl(url)
#     file = folder + '/' + ticker + '.csv'
#     dailyData.to_csv(file)
#     print('daily stock price for [' + ticker + '] got')

def fetchTickerList():    
    print('getting ticker list...')

    nasdaqUrl = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download'
    tickersRawData = getDataframeFromUrl(nasdaqUrl)

    amexUrl = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download'
    tickersRawData = tickersRawData.append(getDataframeFromUrl(amexUrl), ignore_index=True)

    nyseUrl = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download'
    tickersRawData = tickersRawData.append(getDataframeFromUrl(nyseUrl), ignore_index=True)

    print('filtering out tickers...')
    tickersRawData = tickersRawData.dropna(subset=['MarketCap'])

    filter = tickersRawData['MarketCap'].str.contains('B')
    tickersRawData = tickersRawData[filter]

    tickersRawData['MarketCap'] = tickersRawData['MarketCap'].map(lambda x: x.lstrip('$').rstrip('B'))
    tickersRawData['MarketCap'] = tickersRawData['MarketCap'].astype(float)

    tickersRawData = tickersRawData.sort_values(by = "MarketCap", ascending = False)

    # filter = tickersRawData['MarketCap'] > 100.0
    # tickersRawData = tickersRawData[filter]

    filePath = os.path.dirname(os.path.abspath(__file__)) + '/data/tickerList.csv'
    print('saving tickers to ' + filePath)
    tickersRawData.to_csv(filePath)

    print('tickers list done')
    return tickersRawData['Symbol'].to_list()

fetchTickerList()
