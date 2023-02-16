#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
#%%
input_data_folder = 'input_data/'
#%%
calendar = pd.read_csv(f'{input_data_folder}calendar.csv')
# %%
sales = pd.read_csv(f'{input_data_folder}sales_train_validation.csv')
# %%
sell_prices = pd.read_csv(f'{input_data_folder}sell_prices.csv')

calendar = reduce_mem_usage(calendar)
sales = reduce_mem_usage(sales)
sell_prices = reduce_mem_usage(sell_prices)
#%%
sales_train_val = pd.melt(sales,
                        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                        var_name = 'day', value_name = 'demand')
sales_train_val.head(3)
# %%
sales_cal = sales_train_val.merge(calendar[['date', 'wm_yr_wk', 'd']], how='left', left_on='day', right_on = 'd')
# %%
combined = sales_cal.merge(sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
# %%
combined = combined[combined.demand != 0]
# combined.drop('d', axis=1, inplace=True)
combined = combined[['date', 'wm_yr_wk', 'id', 'item_id',
                    'demand', 'sell_price', 'dept_id', 'cat_id', 'store_id', 'state_id', 'day']]
# %%
combined.head()
#%%
# combined.to_csv(f'{input_data_folder}combined_df.csv')
# %%
print('Writing a file to ./data')
combined.to_csv('./data/combined_result.csv')