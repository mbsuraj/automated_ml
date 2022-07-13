from src.load import loader_adaptor_factory as laf
from data_catalog.catalog import catalog
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from scipy.stats import chi2_contingency
from sklearn import preprocessing
import pandas as pd
import datawig
import numpy as np

def main():
    train_data_file = catalog['landing/train']
    test_data_file = catalog['landing/test']
    # Load train and test data
    data_loader_adaptor = laf.LoaderAdaptorFactory.GetLoaderAdaptor(data_file=train_data_file)
    train_data_loader = data_loader_adaptor(data_file=train_data_file)
    test_data_loader = data_loader_adaptor(data_file=test_data_file)
    train_df = train_data_loader.load_data()
    train_df = common_clean_for_all(train_df)
    # Separate y_train from train_data
    y_train = train_df['SalePrice']
    y_train.to_csv(catalog['clean/y'], index=False)
    train_data = train_df[[c for c in train_df.columns if c != 'SalePrice']]
    test_data = test_data_loader.load_data()
    # concatenate by rows - test_data into train_data to form train_data for cleaning purposes
    # train_data = pd.concat([train_data, test_data])
    # start the cleaning process for the entire data
    # separate numeric and categorical data
    # numeric_train = train_data.select_dtypes(include=np.number).copy()
    numeric_cols = ['Id', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
    numeric_train = train_data[numeric_cols]
    numeric_test = test_data[numeric_cols]

    # categorical_train = pd.concat([train_data.select_dtypes(exclude=np.number), train_data['MSSubClass']], axis=1)
    category_cols = ['Id', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition', 'MSSubClass']
    categorical_train = train_data[category_cols]
    categorical_test = test_data[category_cols]

    # numeric
    numeric_train, numeric_test = clean_numeric_data(numeric_train, numeric_test)
    # category
    cat_cramers_v_matrix = cat_corr(categorical_train)
    categorical_train, categorical_test = fill_missing_values_as_notpresent(categorical_train, categorical_test)
    categorical_train, category_test = clean_category_data(categorical_train, categorical_test)
    ordinal_col = ['LotShape', 'ExterQual',
                   'ExterCond','BsmtQual',
                   'BsmtCond', 'BsmtExposure',
                   'HeatingQC', 'KitchenQual',
                   'FireplaceQu', 'GarageQual',
                   'GarageCond', 'PoolQC',
                   'LandSlope', 'BsmtFinType1',
                   'BsmtFinType2', 'Functional',
                   'GarageFinish', 'PavedDrive',
                   'Fence', 'SaleCondition'
                   ]
    ordinal_set = {
        'set1': {
            'order': ['IR3', 'IR2', 'IR1', 'Reg'],
            'set': ['LotShape']
        },
        'set2': {
            'order': ['not present', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
            'set': ['ExterQual', 'ExterCond','BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
        'GarageCond', 'PoolQC']
        },
        'set2a': {
            'order': ['not present', 'No', 'Mn', 'Av', 'Gd'],
            'set': ['BsmtExposure']
        },
        'set3': {
            'order': ['Sev', 'Mod', 'Gtl'],
            'set': ['LandSlope']
        },
        'set4': {
            'order': ['not present', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
            'set': ['BsmtFinType1', 'BsmtFinType2']
        },
        'set5': {
            'order': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ', 'no information'],
            'set': ['Functional']
        },
        'set6': {
            'order': ['not present', 'Unf', 'RFn', 'Fin'],
            'set': ['GarageFinish']
        },
        'set7': {
            'order': ['N', 'P', 'Y'],
            'set': ['PavedDrive']
        },
        'set8': {
            'order': ['not present', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
            'set': ['Fence']
        },
        'set9': {
            'order': ['Partial', 'Family', 'Alloca', 'AdjLand', 'Abnorml', 'Normal'],
            'set': ['SaleCondition']
        }
    }
    nominal_col = [c for c in category_cols if c not in ordinal_col+['Id']]
    categorical_train, categorical_test = encode_categories(category_train=categorical_train,
                                                            category_test=categorical_test,
                                                            ordinal_set=ordinal_set,
                                                            nominal_cols=nominal_col)

    train_data = pd.merge(numeric_train, categorical_train, how='inner', on='Id')
    test_data = pd.merge(numeric_test, categorical_test, how='inner', on='Id')

    train_data.sort_values(by="Id", ascending=True, inplace=True)
    test_data.sort_values(by="Id", ascending=True, inplace=True)

    # separate test data from train data
    # test_data = pd.merge(train_data, test_data[['Id']], how='inner', on='Id')
    # train_data = pd.merge(train_data, train_df[['Id']], how='inner', on='Id')

    # output the processed test and train datasets
    train_data.to_csv(catalog['clean/train'], index=False)
    test_data.to_csv(catalog['clean/test'], index=False)
    return 0

def common_clean_for_all(df):
    df = df.drop_duplicates().copy()
    df = df.dropna(axis=0, how='all').copy()
    return df

def clean_numeric_data(numeric_train, numeric_test):
    # clean numeric data
    # # Remove irrelevant data
    numeric_train.drop(columns=['GarageCars'], inplace=True) # this field is highly correlated and doesn't
    numeric_test.drop(columns=['GarageCars'], inplace=True)
    # add much value to house sales prediction
    # numeric_train.info()
    # impute missing values for LotFrontage
    lotfrontage = 'LotFrontage'
    lotfrontage_fields = ["LotArea", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "TotRmsAbvGrd", "LotFrontage"]
    numeric_train, numeric_test = impute_by_regression(lotfrontage, lotfrontage_fields, numeric_train, numeric_test)
    # impute missing values for MasVnrArea
    massvnrarea = 'MasVnrArea'
    massvnrarea_fields = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', "MasVnrArea"]
    numeric_train, numeric_test = impute_by_regression(massvnrarea, massvnrarea_fields, numeric_train, numeric_test)
    # impute missing values for FullBath
    fullbath = 'BsmtFullBath'
    fullbath_fields = ['GrLivArea', 'OverallQual', 'TotRmsAbvGrd', 'YearBuilt', 'BsmtFullBath']
    numeric_train, numeric_test = impute_by_regression(fullbath, fullbath_fields, numeric_train, numeric_test)

    def impute_missing_values(numeric_data):
        # impute missing values
        numeric_data['BsmtFinSF1'] = numeric_data['BsmtFinSF1'].fillna(0)
        numeric_data['BsmtFinSF2'] = numeric_data['BsmtFinSF2'].fillna(0)
        numeric_data['BsmtUnfSF'] = numeric_data['BsmtUnfSF'].fillna(0)
        numeric_data['TotalBsmtSF'] = numeric_data['TotalBsmtSF'].fillna(0)
        numeric_data['BsmtFullBath'] = numeric_data['BsmtFullBath'].fillna(0)
        numeric_data['BsmtHalfBath'] = numeric_data['BsmtHalfBath'].fillna(0)
        numeric_data['GarageYrBlt'] = numeric_data['GarageYrBlt'].fillna(0)
        numeric_data['GarageArea'] = numeric_data['GarageArea'].fillna(0)
        return numeric_data

    numeric_train = impute_missing_values(numeric_train)
    numeric_test = impute_missing_values(numeric_test)
    return numeric_train, numeric_test

def clean_category_data(category_train, category_test):

    # impute missing values for MasVnrType
    masvnrtype = 'Electrical'
    masvnrtype_fields = ['CentralAir', 'BsmtCond', 'GarageQual', 'Electrical']
    category_train, category_test = impute_by_datawig(masvnrtype, masvnrtype_fields, category_train, category_test)
    # impute missing values for MSZoning
    MSZoning = 'MSZoning'
    MSZoning_fields = ['Neighborhood', 'Alley', 'MSSubClass', 'MSZoning']
    category_train, category_test = impute_by_datawig(MSZoning, MSZoning_fields, category_train, category_test)
    # impute missing values for Functional
    Functional = 'Functional'
    Functional_fields = ['ExterCond', 'HeatingQC', 'ExterQual', 'Functional']
    category_train, category_test = impute_by_datawig(Functional, Functional_fields, category_train, category_test)
    # impute missing values for Functional
    SaleType = 'SaleType'
    SaleType_fields = ['SaleCondition', 'ExterQual', 'BsmtQual', 'SaleType']
    category_train, category_test = impute_by_datawig(SaleType, SaleType_fields, category_train, category_test)
    # Kitchen quality, Exterior1st, Exterior2nd, Utilities
    return category_train, category_test

def impute_by_regression(y_field, x_fields, numeric_train, numeric_test):
    def get_data(numeric_data):
        data = numeric_data[x_fields]
        data_to_impute = data[data[y_field].isnull()]
        data.dropna(inplace=True)
        y = data[y_field]
        X = data.drop(y_field, axis=1)
        X_for_inpute = data_to_impute.drop(y_field, axis=1)
        return y, X, X_for_inpute

    def get_impute_regression_model(X_train, y_train):
        model = LinearRegression()
        model = model.fit(X_train, y_train)
        return model

    def impute(numeric_data, model, X_for_inpute):
        # numeric_train[numeric_train['LotFrontage'].isnull()]['LotFrontage'] = model.predict(X_test)
        numeric_not_null = numeric_data[numeric_data[y_field].isnull()==False]
        numeric_null = numeric_data[numeric_data[y_field].isnull()]
        if numeric_null.shape[0] > 0:
            numeric_null[y_field] = model.predict(X_for_inpute)
            numeric_data = pd.concat([numeric_not_null, numeric_null], axis=0).copy()
            return numeric_data
        else:
            return numeric_data
    y_train, X_train, X_train_for_inpute = get_data(numeric_train)
    y_test, X_test, X_test_for_inpute = get_data(numeric_test)
    model = get_impute_regression_model(X_train, y_train)
    numeric_train = impute(numeric_train, model, X_train_for_inpute)
    numeric_test = impute(numeric_test, model, X_test_for_inpute)
    return numeric_train, numeric_test

def cat_corr(categorical_train):
    data = categorical_train[
        [i for i in categorical_train.columns if i not in ('Id')]]

    label = preprocessing.LabelEncoder()
    data_encoded = pd.DataFrame()

    for i in data.columns:
        data_encoded[i] = label.fit_transform(data[i])

    rows = []

    for var1 in data_encoded:
        col = []
        for var2 in data_encoded:
            cramers = cramers_v(pd.crosstab(data_encoded[var1], data_encoded[var2]).values)  # Cramer's V test
            col.append(round(cramers, 2))  # Keeping of the rounded value of the Cramer's V
        rows.append(col)

    cramers_results = np.array(rows)
    df = pd.DataFrame(cramers_results, columns=data_encoded.columns, index=data_encoded.columns)
    return df

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def fill_missing_values_as_notpresent(category_train,  category_test):
    # impute missing values
    values = {
        "Alley": "not present",
        "MasVnrType": "not present",
        "BsmtQual": "not present",
        "BsmtCond": "not present",
        "BsmtExposure": "not present",
        "BsmtFinType1": "not present",
        "BsmtFinType2": "not present",
        "FireplaceQu": "not present",
        "Functional": "no information",
        "GarageType": "not present",
        "GarageFinish": "not present",
        "GarageQual": "not present",
        "GarageCond": "not present",
        "PoolQC": "not present",
        "Fence": "not present",
        "KitchenQual": "not present",
        "MiscFeature": "not present"
        }
    category_train = category_train.fillna(value=values)
    category_test = category_test.fillna(value=values)
    return category_train, category_test

def impute_by_datawig(y_field, x_fields, category_train, category_test):
    df_train = category_train.dropna()
    df_train_to_impute = category_train[category_train[y_field].isnull()]
    df_test_to_impute = category_test[category_test[y_field].isnull()]
    # Initialize a SimpleImputer model
    imputer = datawig.SimpleImputer(
        input_columns=x_fields,
        # column(s) containing information about the column we want to impute
        output_column=y_field,  # the column we'd like to impute values for
        output_path='imputer_model'  # stores model data and metrics
    )
    # Fit an imputer model on the train data
    imputer.fit(train_df=df_train, num_epochs=10)

    def impute(category_data, imputer, df_to_impute):
        # Impute missing values and return original dataframe with predictions
        category_not_null = category_data[category_data[y_field].isnull() == False]
        category_null = category_data[category_data[y_field].isnull()]
        if df_to_impute.shape[0] >0:
            category_null[y_field] = imputer.predict(df_to_impute)[y_field + '_imputed']
            category_data = pd.concat([category_not_null, category_null], axis=0).copy()
            return category_data
        else:
            return category_data

    category_train = impute(category_train, imputer, df_train_to_impute)
    category_test = impute(category_test, imputer, df_test_to_impute)
    return category_train, category_test

def encode_categories(category_train, category_test, ordinal_set, nominal_cols):

    if ordinal_set:
        # ordinal encoding for ordinal_cols
        for set_name, set_detail in ordinal_set.items():
            order = set_detail['order']
            categories = [order for x in set_detail['set']]
            oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
            # category_train[set_detail['set']] = category_train[set_detail['set']].apply(oe.fit_transform)
            oe.fit(category_train[set_detail['set']])
            category_train[set_detail['set']] = oe.transform(category_train[set_detail['set']])
            category_test[set_detail['set']] = oe.transform(category_test[set_detail['set']])

    if nominal_cols:
        # One Hot encoding for nominal_cols
        ohe = OneHotEncoder(categories='auto', sparse=False, drop='first', handle_unknown='ignore')
        ohe_transformer = ohe.fit(category_train[nominal_cols])
        feature_names = ohe_transformer.get_feature_names_out()
        # for category train
        array_hot_encoded_cattrain = ohe_transformer.transform(category_train[nominal_cols])
        category_train[feature_names] = pd.DataFrame(array_hot_encoded_cattrain, index=category_train.index)
        category_train = category_train.drop(columns=nominal_cols)
        # for category test
        array_hot_encoded_cattest = ohe_transformer.transform(category_test[nominal_cols])
        category_test[feature_names] = pd.DataFrame(array_hot_encoded_cattest, index=category_test.index)
        category_test = category_test.drop(columns=nominal_cols)
    return category_train, category_test

if __name__ == "__main__":
    main()
