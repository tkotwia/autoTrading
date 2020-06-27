import pandas
import os
import sys
from glob import glob
import threading

fullpaths = []
for root, dirs, files in os.walk('/home/gene/git/autoTrading/demo/data'):
    for name in files:
        if ('.txt' in name):
            fullpaths.append(os.path.abspath(os.path.join(root, name)))

# fullpaths = fullpaths[0:10]
threadcount = 10

def iscandidate(path):
    try:
        df = pandas.read_csv(path)
    except:
        return False
    if len(df.index) > 100 and (int)(df['<VOL>'].iloc[-1]) > (int)(df['<VOL>'].iloc[-2]) + (int)(df['<VOL>'].iloc[-3]):
        return True
    return False

def job(id):
    size = (int)(len(fullpaths) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        fullpaths_t = fullpaths[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        fullpaths_t = fullpaths[size * id : size * (id + 1)]
    for fullpath in fullpaths_t:
        if iscandidate(fullpath):
            head, tail = os.path.split(fullpath)
            symbol = tail[0: tail.find('.us.txt')]
            command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/strategy1.py --no-log --source 'local' --symbol '%s'" % symbol
            os.system(command)

threads = []
for i in range(threadcount):
    threads.append(threading.Thread(target = job, args = (i,)))
    threads[i].start()

for t in threads:
    t.join()
