import re

def replace_title_1(title):
    return title.replace('are', '***')

def replace_title_2(title):
    return re.sub(r'((m|M)achine( |-)(l|L)earning)', '', title)