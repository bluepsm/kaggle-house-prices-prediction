import pandas as pd
import joblib

from utils.house_prices.dataset import *


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

random_seed = 42

train = pd.read_csv('datasets/house_prices/train.csv', index_col='Id')
test = pd.read_csv('datasets/house_prices/test.csv', index_col='Id')

imputed_train = imputing_data(train)
imputed_test = imputing_data(test)
full_dataset = pd.concat([imputed_train, imputed_test], ignore_index=True).copy()
target = full_dataset['SalePrice']
full_dataset = full_dataset.drop('SalePrice', axis=1)

model = build_svr_model(full_dataset)
# filename = 'saved_models/svr_model_lot_frontage.sav'
# joblib.dump(model, filename)

lot_frontage_imputed_train = imputing_lot_frontage(model, full_dataset, train)
lot_frontage_imputed_test = imputing_lot_frontage(model, full_dataset, test)

lot_frontage_imputed_train.to_csv('datasets/house_prices/all_data_imputed_train.csv')
lot_frontage_imputed_test.to_csv('datasets/house_prices/all_data_imputed_test.csv')

print(lot_frontage_imputed_train.info())
print(lot_frontage_imputed_test.info())
