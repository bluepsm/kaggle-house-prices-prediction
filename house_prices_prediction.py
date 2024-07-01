import pandas as pd
import ydf
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

random_seed = 42

dataset = pd.read_csv('datasets/house_prices/all_data_imputed_train.csv', index_col='Id')
pred_dataset = pd.read_csv('datasets/house_prices/all_data_imputed_test.csv', index_col='Id')

dataset = dataset.sample(frac=1)
split_idx = len(dataset) * 7 // 10
train_dataset = dataset.iloc[:split_idx]
test_dataset = dataset.iloc[split_idx:]

# print(dataset['SalePrice'].describe())
# # plt.figure()
# sns.displot(dataset['SalePrice'], bins=100)
# plt.show()

# df_num = dataset.select_dtypes(include = ['float64', 'int64'])
# df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# plt.show()

model = ydf.RandomForestLearner(label='SalePrice', task=ydf.Task.REGRESSION).train(train_dataset)
# plot = model.plot_tree()
# plot.to_file('model_tree.html')

evaluation = model.evaluate(test_dataset)
print(evaluation)

# print(model.variable_importances())

# pred = model.predict(pred_dataset)
# pred_df = pd.DataFrame(pred, columns=['SalePrice'])
# print(pred_df)

submission_df = pd.read_csv('datasets/house_prices/sample_submission.csv')
submission_df['SalePrice'] = model.predict(pred_dataset)
submission_df.to_csv('datasets/house_prices/submission.csv', index=False)

