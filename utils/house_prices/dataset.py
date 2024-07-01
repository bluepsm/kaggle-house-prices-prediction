import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


def imputing_data(data):
    data.fillna({'Alley': 'NoAlley', 'MasVnrType': 'NoMasVnr', 'BsmtQual': 'NoBsmt', 'BsmtCond': 'NoBsmt', 'BsmtExposure': 'NoBsmt',
                 'BsmtFinType1': 'NoBsmt', 'BsmtFinType2': 'NoBsmt', 'FireplaceQu': 'NoFrplce', 'GarageType': 'NoGarage',
                 'GarageFinish': 'NoGarage',
                 'GarageQual': 'NoGarage', 'GarageCond': 'NoGarage', 'PoolQC': 'NoPool', 'Fence': 'NoFence', 'MiscFeature': 'NoMisc'},
                inplace=True)

    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode().iloc[0])
    data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode().iloc[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode().iloc[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode().iloc[0])
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode().iloc[0])
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode().iloc[0])
    data['Functional'] = data['Functional'].fillna(data['Functional'].mode().iloc[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode().iloc[0])

    data['MasVnrArea'] = data['MasVnrArea'].fillna(
        data.apply(
            lambda row: 0 if (row['MasVnrType'] == 'NoMasVnr') else data['MasVnrArea'].mean(),
            axis=1
        )
    )
    data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['BsmtFinSF1'].mean(),
            axis=1
        )
    )
    data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['BsmtFinSF2'].mean(),
            axis=1
        )
    )
    data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['BsmtUnfSF'].mean(),
            axis=1
        )
    )
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['TotalBsmtSF'].mean(),
            axis=1
        )
    )
    data['BsmtFullBath'] = data['BsmtFullBath'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['BsmtFullBath'].mean(),
            axis=1
        )
    )
    data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(
        data.apply(
            lambda row: 0 if (row['BsmtQual'] == 'NoBsmt') else data['BsmtHalfBath'].mean(),
            axis=1
        )
    )
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(
        data.apply(
            lambda row: 0 if (row['GarageType'] == 'NoGarage') else data['GarageYrBlt'].mean(),
            axis=1
        )
    )
    data['GarageCars'] = data['GarageCars'].fillna(
        data.apply(
            lambda row: 0 if (row['GarageType'] == 'NoGarage') else data['GarageCars'].mean(),
            axis=1
        )
    )
    data['GarageArea'] = data['GarageArea'].fillna(
        data.apply(
            lambda row: 0 if (row['GarageType'] == 'NoGarage') else data['GarageArea'].mean(),
            axis=1
        )
    )

    return data


def build_svr_model(data, random_seed=42):
    train = data[~data.LotFrontage.isnull()]
    test = data[data.LotFrontage.isnull()]
    # target = train.LotFrontage
    print("LotFrontage has {:} missing value, and {:} values available.".format(test.shape[0], train.shape[0]))

    y_data = train['LotFrontage']
    x_data = train.loc[:, ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood',
                           'Condition1', 'Condition2', 'GarageCars']]

    x_data = pd.get_dummies(x_data)
    x_data = (x_data - x_data.mean()) / x_data.std()
    x_data = x_data.fillna(0)

    clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)
    acc = 0
    acc1 = 0
    acc2 = 0

    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

    for trn, tst in kf.split(train):
        fold_train_samples = train.iloc[trn]
        fold_test_samples = train.iloc[tst]
        neigh_means = fold_train_samples.groupby('Neighborhood')['LotFrontage'].mean()
        all_means = fold_train_samples['LotFrontage'].mean()
        y_pred_neigh_means = fold_test_samples.join(neigh_means, on='Neighborhood', lsuffix='benchmark')['LotFrontage']
        y_pred_all_means = [all_means] * fold_test_samples.shape[0]

        # u1 = ((fold_test_samples['LotFrontage'] - y_pred_neigh_means) ** 2).sum()
        # u2 = ((fold_test_samples['LotFrontage'] - y_pred_all_means) ** 2).sum()
        # v = ((fold_test_samples['LotFrontage'] - fold_test_samples['LotFrontage'].mean()) ** 2).sum()

        clf.fit(x_data.iloc[trn], y_data.iloc[trn])

        acc = acc + mean_absolute_error(fold_test_samples['LotFrontage'], clf.predict(x_data.iloc[tst]))
        acc1 = acc1 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_neigh_means)
        acc2 = acc2 + mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_all_means)

    print('10-Fold Validation Mean Absolute Error results:')
    print('\tSVR: {:.3}'.format(acc / 10))
    print('\tSingle mean: {:.3}'.format(acc2 / 10))
    print('\tNeighbourhood mean: {:.3}'.format(acc1 / 10))

    return clf


def imputing_lot_frontage(model, all_data, data):
    # Columns reference for model
    exists_lot_frontage_rows = all_data[~all_data.LotFrontage.isnull()]
    exists_x = exists_lot_frontage_rows.loc[:, ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType',
                                                'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]
    exists_x = pd.get_dummies(exists_x)
    exists_x = (exists_x - exists_x.mean()) / exists_x.std()
    exists_x = exists_x.fillna(0)

    empty_lot_frontage_rows = data[data.LotFrontage.isnull()]
    empty_x = empty_lot_frontage_rows.loc[:, ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType',
                                              'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']]
    empty_x = pd.get_dummies(empty_x)
    empty_x = (empty_x - empty_x.mean()) / empty_x.std()
    empty_x = empty_x.fillna(0)

    for col in (set(exists_x.columns) - set(empty_x.columns)):
        empty_x[col] = 0

    empty_x = empty_x[exists_x.columns]

    data.loc[data.LotFrontage.isnull(), 'LotFrontage'] = model.predict(empty_x)

    return data
