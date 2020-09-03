from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse
import statistics
from sklearn.linear_model import LinearRegression
import numpy
import matplotlib.pyplot as plt

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('stoplimit', 5),
        ('years', 3),
        ('graph', 'on')
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.calculate_days = 253 * self.p.years
        self.data = numpy.array([])
        self.close = numpy.array([])
        self.lr = numpy.array([])
        self.dates = numpy.array([])

    def notify_order(self, order):
        pass

    def notify_trade(self, trade):
        pass
    
    def next(self):
        if len(self.data) == self.calculate_days:
            model = LinearRegression(fit_intercept=True)
            x = numpy.array(range(self.calculate_days))
            model.fit(x[:, numpy.newaxis], self.data)
            p = model.predict(numpy.array(self.calculate_days + 1).reshape(-1, 1))
            self.close = numpy.append(self.close, self.dataclose[0])
            self.lr = numpy.append(self.lr, p)
            self.dates = numpy.append(self.dates, self.datas[0].datetime.date(0))

            self.data = numpy.delete(self.data, 0, axis = 0)
        
        self.data = numpy.append(self.data, [self.dataclose[0]], axis = 0)

    def stop(self):
        total = 0
        max_diff = 0
        min_diff = 0
        for i in range(len(self.dates)):
            diff = (self.close[i] - self.lr[i]) / self.lr[i]
            if diff > max_diff:
                max_diff = diff
            if diff < min_diff:
                min_diff = diff
            if diff < 0:
                diff = diff * -1
            total += diff
        avg_diff = total / len(self.dates)
        price_upper = self.lr[-1] * (1 + avg_diff)
        price_lower = self.lr[-1] * (1 - avg_diff)

        if self.close[-1] < self.lr[-1]:
            print(self.getdatanames()[0])
        if self.close[-1] < price_lower:
            print(self.getdatanames()[0], "low")
        print('%s, %s, close:%.2f expected: %.2f high:%.2f low:%.2f diff:%.4f min_diff:%.4f max_diff:%.4f' % (self.datas[0].datetime.date(0).isoformat(), self.getdatanames()[0], self.close[-1], self.lr[-1], price_upper, price_lower, avg_diff, min_diff, max_diff))

        if self.p.graph == 'on':
            plt.plot(self.dates, self.close, 'r', ms = 1)
            plt.plot(self.dates, self.lr, 'k')
            plt.plot(self.dates, self.lr * (1 + avg_diff), 'b')
            plt.plot(self.dates, self.lr * (1 - avg_diff), 'b')
            plt.plot(self.dates, self.lr * (1 + max_diff), 'g')
            plt.plot(self.dates, self.lr * (1 + min_diff), 'g')
            plt.show()

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--symbol', required=False, default='MSFT', help='Symbol to backtest')

    parser.add_argument('--graph', required=False, default='on', help = 'plot on/off')

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

    strats = cerebro.addstrategy(TestStrategy, years = 3, graph = args.graph)

    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    if args.source == 'yahoo':
        # Create a Data Feed
        data = bt.feeds.YahooFinanceData(
            dataname=args.symbol,
            fromdate=start,
            todate=end,
            reverse=False)
    elif args.source == 'local':
        data = load_generic_data(args.symbol, start, end)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    cerebro.run(maxcpus=1)
    # try:
    #     # Run over everything
    #     cerebro.run(maxcpus=1)
    # except Exception as e:
    #     print(args.symbol, e)

    # cerebro.plot()
