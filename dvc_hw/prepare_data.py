#%%
import gc
import numpy as np
import pandas as pd

#%%
input_data_folder = 'input_data/'
#%%
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: 
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# %%
sales = pd.read_csv(f'{input_data_folder}sales_train_validation.csv')
sales = reduce_mem_usage(sales)

#%%
sales = pd.melt(sales,
                id_vars = ['id', 'item_id', 'cat_id', 'dept_id', 'store_id', 'state_id'], 
                var_name = 'day', value_name = 'demand')
sales = sales[sales.demand != 0]
print('Data melted')

#%%
calendar = pd.read_csv(f'{input_data_folder}calendar.csv')
calendar = reduce_mem_usage(calendar)
#%%
sales = sales.merge(calendar[['wm_yr_wk', 'd']], how='left', left_on='day', right_on = 'd') #
print('Data merged with calendar')
del calendar
gc.collect()

#%%
sell_prices = pd.read_csv(f'{input_data_folder}sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)
#%%
sales = sales.merge(sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
print('Data merged with prices')
del sell_prices
gc.collect()

#%%
sales = sales[['wm_yr_wk', 'id', 'item_id',
                'demand', 'sell_price', 'cat_id', 'store_id', 'state_id', 'day']]

#%%
print('Writing a file to ./output_data')

half_floats = sales.select_dtypes(include="float16")
sales[half_floats.columns] = half_floats.astype("float32")
sales.to_parquet('output_data/combined_result', partition_cols='cat_id')
#%%
