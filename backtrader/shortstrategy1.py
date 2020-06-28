from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import argparse

# Import the backtrader platform
import backtrader as bt

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('stoplimit', 3),
        ('maxage', 2),
        ('printlog', True)
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datavolume = self.datas[0].volume

        # To keep track of pending orders and buy price/commission
        self.stoporder = []
        self.age = 0
        self.stoplimit = self.p.stoplimit / 100
        self.history = []
        self.winexception = 0
        self.lossexception = 0
        self.lastbuydate = None

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
            if order.issell():
                self.log(
                    'SELL EXECUTED, Price: %.2f, Cost: %.2f' %
                    (order.executed.price,
                    order.executed.value))

                self.age = 0

                limit = order.executed.price * (1 + self.stoplimit)
                stop = order.executed.price * (1 - self.stoplimit)
                self.log('BUY CREATE limit %.2f stop %.2f'%  (limit, stop))
                self.stoporder.append(self.buy(exectype = bt.Order.Limit, price=limit, valid = self.datas[0].datetime.date(0) + datetime.timedelta(days=self.p.maxage), transmit=False))
                self.stoporder.append(self.buy(exectype = bt.Order.StopLimit, price=stop, parent=self.stoporder[0],valid = self.datas[0].datetime.date(0) + datetime.timedelta(days=self.p.maxage), transmit=True))
                return
            else:  # BUY
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f' %
                        (order.executed.price,
                        order.executed.value))  
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

        percentage = trade.pnl / trade.price
        self.log('OPERATION PROFIT, GROSS %.2f, percentage %.2f' %
                 (trade.pnl, percentage))
        self.history.append(percentage)

    def next(self):
        # Check if we are in the market
        if not self.position:
            if self.datavolume[0] < self.datavolume[-1]:
                if ((self.dataclose[0] - self.dataclose[-1]) / self.dataclose[-1]) > self.stoplimit:

                    self.log('SELL CREATE, price %.2f ' % (self.dataclose[0]))
                    self.sell()
                    self.lastbuydate = self.datas[0].datetime.date(0)

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
        history = ""
        wincount = 0
        losscount = 0
        tradecount = 0
        for i in self.history:
            history += " %.1f" % (i * 100)
            if i > 0:
                wincount += 1
                tradecount += 1
            elif i < 0:
                losscount += 1
                tradecount += 1
        self.log(history)

        if not self.params.printlog:
            if self.datas[0].datetime.date(0) == self.lastbuydate:
                self.log('%s, %.2f, %d/%d/%d, %d/%d, %d' % (self.getdatanames()[0], self.kelly(), wincount, losscount, tradecount, self.winexception, self.lossexception, self.dataclose[0] * self.datavolume[0] / 1000000), doprint=True)
        else:
            self.log('%s Stoplimit %.2f maxAge %d Kelly %.2f win %d loss %d total %d winexception %d lossexception %d tradevalue %d' % (self.getdatanames()[0], self.stoplimit, self.p.maxage, self.kelly(), wincount, losscount, tradecount, self.winexception, self.lossexception, self.dataclose[0] * self.datavolume[0] / 1000000), doprint=True)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--symbol', required=False, default='MSFT', help='Symbol to backtest')

    parser.add_argument('--log', dest='printlog', action='store_true', help = 'enable log')
    parser.add_argument('--no-log', dest='printlog', action='store_false', help = 'disable log')
    parser.set_defaults(printlog=True)

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

    strats = cerebro.addstrategy(TestStrategy, printlog = args.printlog)

    if args.source == 'yahoo':
        # Create a Data Feed
        data = bt.feeds.YahooFinanceData(
            dataname=args.symbol,
            fromdate=datetime.datetime(1990, 1, 1),
            todate=datetime.datetime.now(),
            reverse=False)
    elif args.source == 'local':
        data = load_generic_data(args.symbol, datetime.datetime(1990, 1, 1), datetime.datetime.now())

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)

    # cerebro.plot()
