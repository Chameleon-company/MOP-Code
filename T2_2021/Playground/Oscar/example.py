import sys 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
sys.path.append('/Users/wuyoscar/Documents/Project/Playground')

import deakin
table, client = deakin.top_table(top=20)
#table= deakin.keyword_search('parking')
print(table)