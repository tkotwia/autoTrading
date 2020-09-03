from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse
import statistics

from cnn import Cnn

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('printlog', 'full'),
        ('train_end', datetime.datetime(2017, 1, 1))
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog == 'full' or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        #initialize cnns
        self.cnns = []
        for i in range(1):
            filename = '/home/gene/git/autoTrading/backtrader/model/' + self.getdatanames()[0] + str(i) + '.h5'
            self.cnns.append(Cnn(filename))
        # Add a MovingAverageSimple indicator
        self.rsi60 = bt.indicators.RSI_Safe(self.datas[0], period=100)
        self.history = []
        self.holding_days = []
        
        self.indicator_count = 22
        self.indicators = []
        for i in range(self.indicator_count):
            self.indicators.append([])

        for i in range(6, 21):
            j = 0
            self.indicators[j].append(bt.indicators.RSI_Safe(self.datas[0], period=i)) # momentum
            j += 1
            self.indicators[j].append(bt.indicators.WilliamsR(self.datas[0], period=i)) # momentum
            j += 1
            self.indicators[j].append(bt.talib.MFI(self.datahigh, self.datalow, self.dataclose, self.datavolume, period=i)) # momentum
            j += 1
            self.indicators[j].append(bt.indicators.RateOfChange(self.datas[0], period=i)) # momentum
            j += 1
            self.indicators[j].append(bt.talib.CMO(self.dataclose, period=i)) # momentum
            j += 1
            self.indicators[j].append(bt.talib.SMA(self.dataclose, period=i))
            j += 1
            self.indicators[j].append(bt.talib.SMA(self.dataopen, period=i))
            j += 1
            self.indicators[j].append(bt.indicators.ExponentialMovingAverage(self.datas[0], period=i))
            j += 1
            self.indicators[j].append(bt.indicators.WeightedMovingAverage(self.datas[0], period=i))
            j += 1
            self.indicators[j].append(bt.indicators.HullMovingAverage(self.datas[0], period=i))
            j += 1
            self.indicators[j].append(bt.indicators.Trix(self.datas[0], period=i)) # trend
            j += 1
            self.indicators[j].append(bt.indicators.CommodityChannelIndex(self.datas[0], period=i)) # trend
            j += 1
            self.indicators[j].append(bt.indicators.DetrendedPriceOscillator(self.datas[0], period=i)) # trend
            j += 1
            self.indicators[j].append(bt.indicators.DirectionalMovementIndex(self.datas[0], period=i)) # trend
            j += 1
            self.indicators[j].append(bt.indicators.BollingerBands(self.datas[0], period=i)) # volatility
            j += 1

            self.indicators[j].append(bt.indicators.PercentagePriceOscillator(self.datas[0], period1=i))
            j += 1
            self.indicators[j].append(bt.indicators.MeanDeviation(self.datas[0], period=i))
            # j += 1
            # self.indicators[j].append(bt.talib.VAR(self.dataclose, period=i, nbdev=1))
            j += 1
            self.indicators[j].append(bt.talib.TRIMA(self.dataclose, period=i))
            j += 1
            self.indicators[j].append(bt.talib.ADXR(self.datahigh, self.datalow, self.dataclose, period=i))
            j += 1
            self.indicators[j].append(bt.talib.AROONOSC(self.datahigh, self.datalow, period=i))
            j += 1
            self.indicators[j].append(bt.talib.ATR(self.datahigh, self.datalow, self.dataclose, period=i))
            j += 1
            self.indicators[j].append(bt.talib.LINEARREG(self.dataopen, period=i))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f' %
                    (order.executed.price,
                    order.executed.value))

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f' %
                        (order.executed.price,
                        order.executed.value))
                
                # self.stoporder.clear()

        elif order.status in [order.Canceled]:
            self.log('Order Canceled')
        elif order.status in [order.Margin]:
            self.log('Order Margin')
        elif order.status in [order.Rejected]:
            self.log('Order Rejected')
        elif order.status in [order.Expired]:
            self.log('Order expired')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        percentage = trade.pnl / trade.price
        self.log('OPERATION PROFIT, GROSS %.2f, percentage %.2f, days %d' %
                (trade.pnl, percentage, self.holding_day))

        self.history.append(percentage)
        self.holding_days.append(self.holding_day)

    def _parse_data(self, index):
        data = []
        for i in range(15):
            for j in range(self.indicator_count):
                data.append(self.indicators[j][i][index])
            data.append(self.dataopen[index - i])
            data.append(self.dataclose[index - i])
            data.append(self.datahigh[index - i])
            data.append(self.datalow[index - i])
        return data

    def _caculate_value(self):
        c = []
        for i in range(0, -11, -1):
            c.append(self.dataopen[i])
        if (c[-1] == max(c)):
            return 1
        if (c[-1] == min(c)):
            return 0
        return 2

    def next(self):
        if self.datas[0].datetime.date(0) <= self.params.train_end.date():
            data = self._parse_data(-11)
            val = self._caculate_value()
            for i in range(1):
                self.cnns[i].add_train_data(data, val)
            
        else:
            for i in range(1):
                if not self.cnns[i].is_trained():
                    self.log('start training...')
                    self.cnns[i].start_traning()

            data = self._parse_data(0)
            count0 = 0
            count1 = 0
            for i in range(1):
                predict_value = self.cnns[i].predict(data)
                if (predict_value == 0):
                    count0 += 1
                if (predict_value == 1):
                    count1 += 1

            if not self.position:
                if (count0 >= 1):
                    self.buy()
                    self.holding_day = 0
            else:
                self.holding_day += 1
                if (count1 >= 1):
                    self.close()

    def stop(self):
        print(self.holding_days)
        print(sum(self.holding_days))
        print(self.history)
        print(sum(self.history))
        result = 1
        for i in self.history:
            result = result * (1 + i)
        print(result)
        with open('/home/gene/git/autoTrading/abc', 'a') as the_file:
            s = str(result) + '\n' 
            the_file.write(s)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--symbol', required=False, default='SPY', help='Symbol to backtest')

    parser.add_argument('--log', required=False, default='full', help = 'log type')

    parser.add_argument('--source', required=False, default='yahoo', help='source type')

    parser.add_argument('--year', required=False, default='2020', help='year to test')

    return parser.parse_args()

def load_generic_data(symbol, start, end):
    filename = symbol.lower() + '.us.txt'
    for root, dirs, files in os.walk('/home/gene/git/autoTrading/demo/data'):
        for name in files:
            if name == filename:
                datapath = os.path.abspath(os.path.join(root, name))
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        fromdate=start,
        todate=end,
        dtformat='%Y%m%d',
        datetime=2,
        open=4,
        high=5,
        low=6,
        close=7,
        volume=8,
        openinterest=9
    )
    return data

if __name__ == '__main__':
    args = parse_args()

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    year = int(args.year)
    start = datetime.datetime(year - 5, 1, 1)
    # end = datetime.datetime.now()
    end = datetime.datetime(year + 1, 1, 1)
    train_end = datetime.datetime(year, 1, 1)

    strats = cerebro.addstrategy(TestStrategy, printlog = args.log, train_end = train_end)

    # start = datetime.datetime(1997, 1, 1)
    # end = datetime.datetime(2017,12,31)
    if args.source == 'yahoo':
        # Create a Data Feed
        data = bt.feeds.YahooFinanceData(
            dataname=args.symbol,
            fromdate=start,
            todate=end,
            reverse=False)
        vix = bt.feeds.YahooFinanceData(
            dataname='^vix',
            fromdate=start,
            todate=end,
            reverse=False)
    elif args.source == 'local':
        try:
            data = load_generic_data(args.symbol, start, end)
            vix = load_generic_data('vix', start, end)
        except Exception as e:
            print(args.symbol, e)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)
    cerebro.adddata(vix)

    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)

    cerebro.broker.set_coc(True)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    cerebro.run(maxcpus=1)
    # try:
    #     # Run over everything
    #     cerebro.run(maxcpus=1)
    # except Exception as e:
    #     print(args.symbol, e)
