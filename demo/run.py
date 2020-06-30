import pandas
import os
import sys
from glob import glob
import threading
import argparse

def getCandidates():
    symbols = []
    try:
        df = pandas.read_csv('/home/gene/git/autoTrading/demo/data/tickerList.csv')
    except:
        pass

    for i in range(2000):
        symbols.append(df['Symbol'].iloc[i])
    return symbols

symbols = getCandidates()
threadcount = 10

def job(id, args):
    size = (int)(len(symbols) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        symbols_t = symbols[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        symbols_t = symbols[size * id : size * (id + 1)]
    for symbol in symbols_t:
        if args.type == 'long':
            command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/longstrategy1.py --log 'csv' --source 'local' --symbol '%s'" % symbol
            os.system(command)
        if args.type == 'short':
            command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/shortstrategy1.py --log 'csv' --source 'local' --symbol '%s'" % symbol
            os.system(command)

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Sample for Order Target')

    parser.add_argument('--type', required=False, default='long', help='Symbol to backtest')
    return parser.parse_args()


args = parse_args()
threads = []
for i in range(threadcount):
    threads.append(threading.Thread(target = job, args = (i, args)))
    threads[i].start()

for t in threads:
    t.join()
