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

    # filter = tickersRawData['MarketCap'].str.contains('B')
    # tickersRawData = tickersRawData[filter]

    # tickersRawData['MarketCap'] = tickersRawData['MarketCap'].map(lambda x: x.lstrip('$').rstrip('B'))
    # tickersRawData['MarketCap'] = tickersRawData['MarketCap'].astype(float)

    # tickersRawData = tickersRawData.sort_values(by = "MarketCap", ascending = False)

    #filter = tickersRawData['MarketCap'] > 500.0
    #tickersRawData = tickersRawData[filter]

    filePath = os.path.dirname(os.path.abspath(__file__)) + '/data/tickerList.csv'
    print('saving tickers to ' + filePath)
    tickersRawData.to_csv(filePath)

    print('tickers list done')
    return tickersRawData['Symbol'].to_list()

# fecth SPY for benchmark
# fetchDailyStockPrice(ticker = 'SPY', folder = os.path.dirname(os.path.abspath(__file__)) + '/data')

# for i, ticker in enumerate(tickers):
#     print('fetching data ', i + 1, '/', len(tickers))
#     fetchDailyStockPrice(ticker = ticker, folder = os.path.dirname(os.path.abspath(__file__)) + '/data')

tickers = fetchTickerList()

candidates = []

threadcount = 8

def job(id):
    candidates[id] = []
    size = (int)(len(tickers) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        tickers_t = tickers[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        tickers_t = tickers[size * id : size * (id + 1)]
    for ticker in tickers_t:
        hist = yfinance.Ticker(ticker).history(period ='5d')
        volumn = hist['Volume']
        close = hist['Close']
        print("Thread %d processing %s" %(id, ticker))
        if (len(volumn) >= 5 and (volumn[2] + volumn[3]) < volumn[4] and (close[4] - close[3]) / close[3] > 0.03):
            candidates[id].append(ticker)
    #    command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/test1.py --symbol '%s'" % ticker
    #    os.system(command)

threads = []
for i in range(threadcount):
    candidates.append([])
    threads.append(threading.Thread(target = job, args = (i,)))

    threads[i].start()

for t in threads:
    t.join()

result = []
for i in range(threadcount):
    result += i

print(result)

