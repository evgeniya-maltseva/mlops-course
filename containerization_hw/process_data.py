#%%
import gc
import numpy as np
import pandas as pd
import dask.dataframe as dd

#%%
input_data_folder = 'input_data/'
#%%
calendar = dd.read_csv(f'{input_data_folder}calendar.csv')
# %%
sales = dd.read_csv(f'{input_data_folder}sales_train_validation.csv')
# %%
sell_prices = dd.read_csv(f'{input_data_folder}sell_prices.csv')

#%%
sales_train_val = dd.melt(sales,
                        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                        var_name = 'day', value_name = 'demand')
# sales_train_val.head(3)
print('Data melted')
del sales
print('sales deleted')
gc.collect()

# %%
sales_cal = sales_train_val.merge(calendar[['date', 'wm_yr_wk', 'd']], how='left', left_on='day', right_on = 'd')
print('Data merged with calendar')
del calendar, sales_train_val
gc.collect()

# %%
combined = sales_cal.merge(sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
print('Data merged with prices')
del sell_prices, sales_cal
gc.collect()

# %%
combined = combined[combined.demand != 0]
# combined.drop('d', axis=1, inplace=True)
combined = combined[['date', 'wm_yr_wk', 'id', 'item_id',
                    'demand', 'sell_price', 'dept_id', 'cat_id', 'store_id', 'state_id', 'day']]

# %%
print('Writing a file to ./output_data')
combined.to_parquet('output_data/combined_result.parquet', partition_on='state_id')

# calendar.to_csv('output_data/calendar_new.csv')
# %%
