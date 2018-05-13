import os

lists = os.listdir('/users/singlestar/desktop/data_stock/')
for item in lists:
    echo = 'iconv -f GBK -t utf8 /users/singlestar/desktop/data_stock/' + item  + '> /users/singlestar/desktop/new/' + item
    os.system(echo)