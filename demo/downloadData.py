import os

os.system('rm /home/gene/git/autoTrading/demo/data/d_us_txt.zip')
os.system('rm -r /home/gene/git/autoTrading/demo/data/data')
os.system('wget -P /home/gene/git/autoTrading/demo/data https://static.stooq.com/db/h/d_us_txt.zip')
os.system('unzip /home/gene/git/autoTrading/demo/data/d_us_txt.zip -d /home/gene/git/autoTrading/demo/data/')