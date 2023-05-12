#%%
import pandas as pd
from ydata_profiling import ProfileReport
import pickle
import lightgbm as lgb
import json
import shap
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

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

categorical_features = ['item_id', 'cat_id']
for col in categorical_features:
    data[col] = data[col].astype('category')

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
                    'n_estimators': 10,
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

#%%

train_mask = df['d']<=END_TRAIN
valid_mask = train_mask&(df['d']>(END_TRAIN-P_HORIZON))

train_data, train_target = df[train_mask][features_columns], df[train_mask][TARGET]
valid_data, valid_target = df[valid_mask][features_columns], df[valid_mask][TARGET]

encoder = OrdinalEncoder()
train_data[categorical_features] = encoder.fit_transform(train_data[categorical_features])
valid_data[categorical_features] = encoder.transform(valid_data[categorical_features])

lgb_train = lgb.Dataset(train_data, label=train_target)
lgb_valid = lgb.Dataset(valid_data, label=valid_target, reference=lgb_train)

estimator = lgb.train(lgb_params,
                        lgb_train,
                        valid_sets = [lgb_valid],
                        verbose_eval = 100,
                        )

model_stats = pd.DataFrame(
    {'name':estimator.feature_name(),
    'imp':estimator.feature_importance()}
    ).sort_values('imp',ascending=False).head(25).to_dict(orient='list')
model_stats['best_rmse'] = estimator.best_score['valid_0']['rmse']

with open('metrics.json', 'w') as f:
    json.dump(model_stats, f)
model_name = model_dir+'lgb_model_'+store_id+'_v'+str(VERSION)+'.pickle'
pickle.dump(estimator, open(model_name, 'wb'))

# %%


explainer = shap.Explainer(estimator)

shap_values = explainer(train_data)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig('shap_waterfall_plot.png', bbox_inches='tight')
# shap.plot(shap_values)
# plt.savefig('shap_waterfall_plot.png')
plt.clf()
# %%
