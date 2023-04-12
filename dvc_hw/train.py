#%%
import pandas as pd
from ydata_profiling import ProfileReport
import pickle
import lightgbm as lgb
import json

#%%
model_dir = 'models/'
#%%
data = pd.read_parquet('output_data/combined_result')
#%%
# profile = ProfileReport(data, title="Profiling Report")
# %%
# profile.to_file("your_report.html")
# %%
# calendar = pd.read_csv('input_data/calendar.csv')

#%%

# data = data.merge(calendar[['d', 'date']], how='left', left_on='day', right_on='d')
# %%
INDEX = ['d', 'id']
TARGET = 'demand'

# %%
features_to_cat = ['id', 'item_id']
for col in features_to_cat:
    data[col] = data[col].astype('category')
# %%
print('Prices')

# We can do some basic aggregations
data['price_max'] = data.groupby(['store_id','item_id'])['sell_price'].transform('max')
data['price_min'] = data.groupby(['store_id','item_id'])['sell_price'].transform('min')
data['price_std'] = data.groupby(['store_id','item_id'])['sell_price'].transform('std')
data['price_mean'] = data.groupby(['store_id','item_id'])['sell_price'].transform('mean')

data['price_norm'] = data['sell_price']/data['price_max']

data['price_momentum'] = data['sell_price']/data.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))

#%%
data['d'] = data['day'].apply(lambda x: x.split('_')[1]).astype(int)
#%% Drop unnecessary columns
# data = data.drop(['store_id', 'state_id'], axis=1)
STORES_IDS = ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']
store_id = 'CA_1'
df = data[data['store_id']==store_id]


lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.015,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 300,
                    'boost_from_average': False,
                    'verbose': -1,
                } 

INDEX = ['d', 'id']
TARGET = 'demand'
START_TRAIN = 0                  
END_TRAIN   = 1913 - 28*2      
P_HORIZON   = 28   
SEED = 42
VERSION = 1

features_columns = ['item_id', 
                    'cat_id', 'price_max', 'price_min', 'price_std',
                    'price_mean', 'price_norm', 'price_momentum']
train_mask = df['d']<=END_TRAIN
valid_mask = train_mask&(df['d']>(END_TRAIN-P_HORIZON))

train_data = lgb.Dataset(df[train_mask][features_columns], 
                    label=df[train_mask][TARGET])

valid_data = lgb.Dataset(df[valid_mask][features_columns], 
                    label=df[valid_mask][TARGET])

estimator = lgb.train(lgb_params,
                        train_data,
                        valid_sets = [valid_data],
                        verbose_eval = 100,
                        )

model_stats = pd.DataFrame(
    {'name':estimator.feature_name(),
    'imp':estimator.feature_importance()}
    ).sort_values('imp',ascending=False).head(25).to_dict(orient='list')
model_stats['best_rmse'] = estimator.best_score['valid_0']['rmse']

with open('metrics_imp.json', 'w') as f:
    json.dump(model_stats, f)
model_name = model_dir+'lgb_model_'+store_id+'_v'+str(VERSION)+'.pickle'
pickle.dump(estimator, open(model_name, 'wb'))

# %%
