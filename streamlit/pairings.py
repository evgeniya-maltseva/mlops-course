#%%
import streamlit as st
import pandas as pd
import numpy as np
import re

#%%

st.title('Pairings comparison')

data_path = "input_data/pairings.txt"

with open(data_path) as tf:
    lines = tf.readlines()
# %%

#%%
s = '2023-03-01 16:28:34,336 [solver-main] Number of pairings [(LED-KZN):(KZN-MRV):(MRV-KZN)>(KZN-MRV):(MRV-KZN):(KZN-LED)] is 60\n'
regex = re.compile('[A-Z]')


re.search(regex, s)
#%%
for l in lines:
    print(l.split('Number of pairings')[-1])
# %%
