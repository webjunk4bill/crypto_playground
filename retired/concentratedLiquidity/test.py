import pickle
import pandas as pd
import re
import numpy as np
import cl_classes as cl
'''
start = pd.Timestamp('2024-05-01')
stop = pd.Timestamp('2024-06-16')

brett_weth_p3 = cl.V3DexLP('0x76Bf0abD20f1e0155Ce40A62615a90A709a6C3D8', 'base',
                               'based-brett', start, stop, 'eth')
'''
loaded_lp = cl.V3DexLP.load_from_pickle('in_data/WETH_0.0005.pkl')
# loaded_lp = brett_weth_p3

my_lp = cl.LiquidityPosition(loaded_lp)
result = my_lp.simulate_range(1000, 0.1, 0.40, True, False)

print(result)
