import inspect
from datetime import datetime

def debug(*msg, sep='\t'):
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print(f'[{time}] File "{file_name}", line {ln}\t', end='')
    for m in msg:
        print(m, end=sep)
    print('')
