import random
from datetime import datetime

def gen_id(basename):
    return basename + str(int(datetime.now().timestamp() * 10))
