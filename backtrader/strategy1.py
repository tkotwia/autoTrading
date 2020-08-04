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
        self.datavolume = self.datas[0].volume

        #initialize cnn
        self.cnn = Cnn()
        # Add a MovingAverageSimple indicator
        self.rsi60 = bt.indicators.RSI_Safe(self.datas[0], period=100)
        self.history = []
        
        self.rsi = []
        self.rsiema = []
        self.williams = []
        self.wma = []
        self.ema = []
        self.sma = []
        self.hma = []
        self.tripleema = []
        self.cci = []
        self.macd = []
        self.ppo = []
        self.roc = []
        self.dmi = []
        self.bb = []
        self.md = []
        for i in range(6, 21):
            self.rsi.append(bt.indicators.RSI_Safe(self.datas[0], period=i))
            self.rsiema.append(bt.indicators.RSI_EMA(self.datas[0], period=i, safediv=True))
            self.williams.append(bt.indicators.WilliamsR(self.datas[0], period=i))
            self.wma.append(bt.indicators.WeightedMovingAverage(self.datas[0], period=i))
            self.ema.append(bt.indicators.ExponentialMovingAverage(self.datas[0], period=i))
            self.sma.append(bt.indicators.SimpleMovingAverage(self.datas[0], period=i))
            self.hma.append(bt.indicators.HullMovingAverage(self.datas[0], period=i))
            self.tripleema.append(bt.indicators.TripleExponentialMovingAverage(self.datas[0], period=i))
            self.cci.append(bt.indicators.CommodityChannelIndex(self.datas[0], period=i))
            self.macd.append(bt.indicators.MACD(self.datas[0], period_me1=i))
            self.ppo.append(bt.indicators.PercentagePriceOscillator(self.datas[0], period1=i))
            self.roc.append(bt.indicators.CommodityChannelIndex(self.datas[0], period=i))
            self.dmi.append(bt.indicators.DirectionalMovementIndex(self.datas[0], period=i))
            self.bb.append(bt.indicators.BollingerBands(self.datas[0], period=i))
            self.md.append(bt.indicators.MeanDeviation(self.datas[0], period=i))
            # self.md.append(bt.indicators.MovingAverageSimple(self.datas[1], period=i))

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
        self.log('OPERATION PROFIT, GROSS %.2f, percentage %.2f' %
                (trade.pnl, percentage))

        self.history.append(percentage)

    def _parse_data(self, index):
        data = []
        for i in range(0, 15):
            data.append(self.rsi[i][index])
            data.append(self.rsiema[i][index])
            data.append(self.williams[i][index])
            data.append(self.wma[i][index])
            data.append(self.ema[i][index])
            data.append(self.sma[i][index])
            data.append(self.hma[i][index])
            data.append(self.tripleema[i][index])
            data.append(self.cci[i][index])
            data.append(self.macd[i][index])
            data.append(self.ppo[i][index])
            data.append(self.roc[i][index])
            data.append(self.dmi[i][index])
            data.append(self.bb[i][index])
            data.append(self.md[i][index])
        return data

    def _caculate_value(self):
        max_diff = -1000
        min_diff = 1000
        for i in range(0, -11, -1):
            diff = (self.dataclose[i] - self.dataclose[-11]) / self.dataclose[-11]
            if (diff > max_diff):
                max_diff = diff
            elif (diff < min_diff):
                min_diff = diff

        if (min_diff > 0):
            return 0
        if (max_diff < 0):
            return 1
        return 2

    def next(self):
        if self.datas[0].datetime.date(0) <= self.params.train_end.date():
            data = self._parse_data(-11)
            val = self._caculate_value()
            self.cnn.add_train_data(data, val)
            
        else:
            if not self.cnn.is_trained():
                self.log('start training...')
                self.cnn.start_traning()

            data = self._parse_data(0)
            predict_value = self.cnn.predict(data)
            if not self.position:
                if (predict_value == 0):
                    self.buy()
            else:
                if predict_value == 1:
                    self.close()

    def stop(self):
        print(self.history)
        print(sum(self.history))
        result = 1
        for i in self.history:
            result = result * (1 + i)
        print(result)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--symbol', required=False, default='SPY', help='Symbol to backtest')

    parser.add_argument('--log', required=False, default='full', help = 'log type')

    parser.add_argument('--source', required=False, default='yahoo', help='source type')

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

    start = datetime.datetime(2007, 1, 1)
    end = datetime.datetime.now()
    train_end = datetime.datetime(2017, 1, 1)

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
