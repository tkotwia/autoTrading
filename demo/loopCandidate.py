import os
import threading

tickers = eval("['AXAS', 'AKER', 'APYX', 'FUV', 'BCYC', 'BOXL', 'BWEN', 'CLRB', 'CJJD', 'CNET', 'CNSP', 'CODX', 'CRIS', 'CYAN', 'CYCN', 'DAIO', 'DTSS', 'DXLG', 'DISCB', 'DOGZ', 'MOHO', 'ELTK', 'ENLV', 'EVSI', 'EVGN', 'XELA', 'FOCS', 'FFHL', 'GNUS', 'GLBS', 'GSMG', 'GRIN', 'GO', 'GHSI', 'HEPA', 'HIHO', 'HUIZ', 'HYRE', 'ICLR', 'IEC', 'IEA', 'INO', 'JOUT', 'KMDA', 'KTOV', 'LSBK', 'LE', 'LTRPB', 'LLEX', 'LMB', 'LLNW', 'LMNL', 'LONE', 'LKCO', 'MICT', 'MTP', 'GRIL', 'NNDM', 'NMCI', 'NCSM', 'NEPT', 'STIM', 'NOVN', 'OFED', 'OCGN', 'ONCY', 'SEED', 'OTIC', 'PENN', 'PSHG', 'PPIH', 'PLUG', 'PFIE', 'PROF', 'PXS', 'RAVE', 'RCON', 'RFIL', 'RELL', 'SHIP', 'SLS', 'SNES', 'SINO', 'SLGL', 'SPI', 'STRM', 'WISA', 'SCON', 'TLGT', 'TOPS', 'TRCH', 'TACT', 'TBIO', 'TC', 'TOUR', 'WIMI', 'YTEN', 'CTIB', 'ZN', 'AE', 'BRN', 'BIOX', 'DXF', 'ENSV', 'HUSA', 'INFU', 'IHT', 'LLEX', 'MTNB', 'NNVC', 'PTN', 'XPL', 'AMRC', 'CEL', 'CEPU', 'XRF', 'CMRE', 'DVD', 'DLNG', 'FSLY', 'GPRK', 'GOL', 'HEQ', 'MHK', 'PFGC', 'QD', 'SQNS', 'EGY', 'IAE', 'EHI', 'HIX', 'YETI', 'DAO', 'ZYME']")

threadcount = 10

def job(id):
    size = (int)(len(tickers) / threadcount)
    if (id == threadcount - 1):
        print("Thread %d from %d to end" % (id, size * id))
        tickers_t = tickers[size * id :]
    else:
        print("Thread %d from %d to %d" % (id, size * id, size * (id + 1)))
        tickers_t = tickers[size * id : size * (id + 1)]
    for ticker in tickers_t:
        command = "/usr/bin/python3 /home/gene/git/autoTrading/backtrader/strategy1.py --no-log --symbol '%s'" % ticker
        os.system(command)

threads = []
for i in range(threadcount):
    threads.append(threading.Thread(target = job, args = (i,)))
    threads[i].start()

for t in threads:
    t.join()

