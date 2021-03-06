from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse
import statistics

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('stoplimit', 3),
        ('printlog', 'full')
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
        self.benchmarkclose = self.datas[1].close

        # To keep track of pending orders and buy price/commission
        self.stoporder = []
        self.age = 0
        self.stoplimit = self.p.stoplimit / 100
        self.history = []
        self.windates = []
        self.lossdates = []
        self.winexception = 0
        self.lossexception = 0
        self.lastbuydate = None
        self.overwritepnl = 0

        # Add a MovingAverageSimple indicator
        self.sma5 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=5)
        self.sma10 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=10)
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=20)

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

                limit = order.executed.price * (1 + self.stoplimit)
                stop = order.executed.price * (1 - self.stoplimit)
                self.log('SELL CREATE limit %.2f stop %.2f'%  (limit, stop))
                self.stoporder.append(self.sell(exectype = bt.Order.StopLimit, price=stop, valid = 0))
                self.stoporder.append(self.sell(exectype = bt.Order.Limit, price=limit, valid = 0, oco=self.stoporder[0]))
                return
            else:  # Sell
                selltype = ""
                if not self.stoporder:
                    selltype = 'close'
                elif order == self.stoporder[0]:
                    selltype = "stop"
                elif order == self.stoporder[1]:
                    selltype = "limit"

                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f %s' %
                        (order.executed.price,
                        order.executed.value, selltype))  
                for o in self.stoporder:
                    if (o.alive()):
                        self.cancel(o)
                self.stoporder.clear()
                self.age = 0


        elif order.status in [order.Canceled]:
            self.log('Order Canceled')
        elif order.status in [order.Margin]:
            self.log('Order Margin')
        elif order.status in [order.Rejected]:
            self.log('Order Rejected')
        elif order.status in [order.Expired]:
            self.log('Order expired')
            if self.stoporder:
                self.log('Close position')
                self.stoporder.clear()
                self.close()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        if self.overwritepnl != 0:
            percentage = self.overwritepnl / trade.price
        else:
            percentage = trade.pnl / trade.price
        self.log('OPERATION PROFIT, GROSS %.2f, percentage %.2f' %
                (trade.pnl, percentage))
        self.history.append(percentage)
        if percentage >= 0.03:
            self.windates.append(self.datas[0].datetime.date(0))
        elif percentage <= -0.03:
            self.lossdates.append(self.datas[0].datetime.date(0))

    def next(self):
        # Check if we are in the market
        if not self.position:
            vols = []
            for i in range(-1, -11, -1):
                vols.append(self.datavolume[i])
            
            mean = statistics.mean(vols)
            stdev = statistics.stdev(vols)
            if self.datavolume[0] > mean + 1.5 * stdev:
                if ((self.dataclose[0] - self.dataclose[-1]) / self.dataclose[-1]) > self.stoplimit and \
                    self.sma5[0] > self.sma20[0]:

                    self.log('BUY CREATE, price %.2f ' % (self.dataclose[0]))
                    # self.buy(exectype = bt.Order.Limit, price=self.dataclose[0], valid = self.datas[0].datetime.date(0) + datetime.timedelta(days=2))
                    self.buy()
                    self.lastbuydate = self.datas[0].datetime.date(0)
                    self.age = 0
                    self.overwritepnl = 0
        else:
            self.age += 1
            if self.age == 1:
                stop = min(self.datas[0].low[-1], self.dataclose[-1] * 0.95)
                if (self.datas[0].low[0] < stop):
                    self.overwritepnl = stop - self.datas[0].open[0]

    def kelly(self):
        history = []
        wincount = 0
        losscount = 0
        totalcount = 0
        for i in self.history:
            if (i > 0.05):
                history.append(0.05)
                self.winexception += 1
            elif i < -0.05:
                history.append(-0.05)
                self.lossexception += 1
            else:
                history.append(i)

            if (i > 0):
                wincount += 1
                totalcount += 1
            elif (i < 0):
                losscount += 1
                totalcount += 1
        
        if totalcount == 0:
            return 0
        if losscount == 0:
            return 1
        if wincount == 0:
            return 0

        p = wincount / totalcount
        q = losscount / totalcount
        
        expect_win = 0
        expect_loss = 0
        for i in history:
            if i >= 0:
                expect_win += i
            if i < 0:
                expect_loss += (i * -1)
        b = (expect_win / wincount) / (expect_loss / losscount)

        if b == 0:
            return 0

        self.log('p %.2f q %.2f b %.2f' % (p, q, b))
        return p - (q / b)
        
    def stop(self):
        wincount = 0
        losscount = 0
        tradecount = 0
        for i in self.history:
            if i > 0:
                wincount += 1
                tradecount += 1
            elif i < 0:
                losscount += 1
                tradecount += 1

        if self.params.printlog == 'full':
            print(["%.2f" % e for e in self.history])
            print([date.strftime('%Y%m%d') for date in self.windates])
            print([date.strftime('%Y%m%d') for date in self.lossdates])

        if self.params.printlog == 'csv':
            if self.datas[0].datetime.date(0) == self.lastbuydate:
                self.log('%s, %.2f, %d/%d/%d, %d/%d, %d' % (self.getdatanames()[0], self.kelly(), wincount, losscount, tradecount, self.winexception, self.lossexception, self.dataclose[0] * self.datavolume[0] / 1000000), doprint=True)
        else:
            if (tradecount == 0):
                winrate = 0
            else:
                winrate = wincount / tradecount

            self.log('%s Stoplimit %.2f Kelly %.2f win %d loss %d total %d winrate %.2f winexception %d lossexception %d tradevalue %d' % (self.getdatanames()[0], self.stoplimit, self.kelly(), wincount, losscount, tradecount, winrate, self.winexception, self.lossexception, self.dataclose[0] * self.datavolume[0] / 1000000), doprint=True)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--symbol', required=False, default='MSFT', help='Symbol to backtest')

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

    # Add a strategy
    # strats = cerebro.optstrategy(
    #     TestStrategy,
    #     stoplimit=range(2, 4))

    strats = cerebro.addstrategy(TestStrategy, printlog = args.log)

    start = datetime.datetime(2005, 1, 1)
    end = datetime.datetime.now()
    if args.source == 'yahoo':
        # Create a Data Feed
        data = bt.feeds.YahooFinanceData(
            dataname=args.symbol,
            fromdate=start,
            todate=end,
            reverse=False)
        benchmark = bt.feeds.YahooFinanceData(
            dataname='vti',
            fromdate=start,
            todate=end,
            reverse=False)
    elif args.source == 'local':
        try:
            data = load_generic_data(args.symbol, start, end)
            benchmark = load_generic_data('vti', start, end)
        except Exception as e:
            print(args.symbol, e)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)
    cerebro.adddata(benchmark)

    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    try:
        # Run over everything
        cerebro.run(maxcpus=1)
    except Exception as e:
        print(args.symbol, e)

    #cerebro.plot()
