import os
import threading

tickers = eval("['VNET', 'IOTS', 'AFMD', 'ALT', 'AMRN', 'AMOV', 'AMRS', 'APDN', 'AQMS', 'ABIO', 'FUV', 'ARKR', 'ASTC', 'ATOS', 'AVT', 'BIOL', 'BMRA', 'BNTX', 'BLNK', 'CNTG', 'CHEK', 'CAAS', 'CCRC', 'CJJD', 'CLEU', 'CORT', 'COWN', 'CRSA', 'EKSO', 'ENTX', 'EVLO', 'EVK', 'EVOK', 'FPRX', 'FRAN', 'HUGE', 'FFHL', 'GNMK', 'GTEC', 'GRNQ', 'HHR', 'HYMC', 'ICON', 'IDEX', 'INVA', 'INSU', 'JAKK', 'KZIA', 'KIRK', 'KTOV', 'LMST', 'LPCN', 'MARA', 'MDGS', 'MTSL', 'MGEN', 'MOTS', 'NAKD', 'NK', 'OSS', 'PHIO', 'PLXP', 'PRPO', 'RIBT', 'RIOT', 'SGBX', 'PIXY', 'SINT', 'SKYS', 'SNGX', 'SNOA', 'SONO', 'STAA', 'STAF', 'MITO', 'WISA', 'SNSS', 'SPRT', 'SNCR', 'TESS', 'PECK', 'THTX', 'THMO', 'TRIB', 'TRUP', 'MEDS', 'USAU', 'USWS', 'XELB', 'XTLB', 'YIN', 'ZVO', 'ALO', 'AWX', 'JOB', 'SIM', 'IHT', 'MMX', 'SNMP', 'SMTS', 'TAT', 'VOLT', 'AAP', 'AMOV', 'ASA', 'BOX', 'BEDU', 'CHRA', 'CWEN', 'CCR', 'CLGX', 'CELP', 'HCR', 'NVTA', 'MNK', 'MODN', 'MYOV', 'JPI', 'ROYT', 'TECK', 'SHLL', 'UBA', 'SPCE']")

threadcount = 8

def job(id):
    size = (int)(len(tickers) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        tickers_t = tickers[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        tickers_t = tickers[size * id : size * (id + 1)]
    for ticker in tickers_t:
        command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/test1.py --symbol '%s'" % ticker
        os.system(command)

threads = []
for i in range(threadcount):
    threads.append(threading.Thread(target = job, args = (i,)))
    threads[i].start()

for t in threads:
    t.join()

