import pandas
import os
import sys
from glob import glob
import threading
import argparse

fullpaths = []
for root, dirs, files in os.walk('/home/gene/git/autoTrading/demo/data'):
    for name in files:
        if ('.txt' in name):
            fullpaths.append(os.path.abspath(os.path.join(root, name)))

# fullpaths = fullpaths[0:10]
threadcount = 10

mintradevalue = 100000000

def islongcandidate(path):
    try:
        df = pandas.read_csv(path)
    except:
        return False
    if len(df.index) > 100 and \
        (int)(df['<VOL>'].iloc[-1]) > (int)(df['<VOL>'].iloc[-2]) + (int)(df['<VOL>'].iloc[-3]) and\
        (int)(df['<VOL>'].iloc[-1]) * (float)(df['<CLOSE>'].iloc[-1]) > mintradevalue:
        return True
    return False

def isshortcandidate(path):
    try:
        df = pandas.read_csv(path)
    except:
        return False
    if len(df.index) > 100 and \
        (int)(df['<VOL>'].iloc[-1]) < (int)(df['<VOL>'].iloc[-2]) and\
        ((float)(df['<CLOSE>'].iloc[-1]) - (float)(df['<CLOSE>'].iloc[-2])) / (float)(df['<CLOSE>'].iloc[-2]) > 0.03 and\
        (int)(df['<VOL>'].iloc[-1]) * (float)(df['<CLOSE>'].iloc[-1]) > mintradevalue:
        return True
    return False

def job(id, args):
    size = (int)(len(fullpaths) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        fullpaths_t = fullpaths[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        fullpaths_t = fullpaths[size * id : size * (id + 1)]
    for fullpath in fullpaths_t:
        if args.type == 'long' and islongcandidate(fullpath):
            head, tail = os.path.split(fullpath)
            symbol = tail[0: tail.find('.us.txt')]
            command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/longstrategy1.py --no-log --source 'local' --symbol '%s'" % symbol
            os.system(command)
        if args.type == 'short' and isshortcandidate(fullpath):
            head, tail = os.path.split(fullpath)
            symbol = tail[0: tail.find('.us.txt')]
            command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/shortstrategy1.py --no-log --source 'local' --symbol '%s'" % symbol
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
