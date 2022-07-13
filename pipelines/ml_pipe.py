from data_catalog.catalog import catalog
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import pandas as pd
import numpy as np

def main():
    # x = sm.add_constant(train_data)
    # mdl = model.get_robustcov_results(cov_type='HAC', maxlags=1)
    # predict the test data
    # _, lr_metrics = lr_prediction()
    # print("Linear Regression Metrics: \n", lr_metrics)
    # _, ridge_metrics = ridge_prediction()
    # print("Ridge Regression Metrics: \n", ridge_metrics)
    # _, lasso_metrics = lasso_prediction()
    # print("Lasso Regression Metrics: \n", lasso_metrics)
    # _, elastinet_metrics = elastinet_prediction()
    # print("Elastinet Regression Metrics: \n", elastinet_metrics)
    _, deeplr_metrics = deeplearning_prediction()
    print("Deep Learning Regression Metrics: \n", deeplr_metrics)



def get_model_data():
    train_data = pd.read_csv(catalog['clean/train'])
    y_train = pd.read_csv(catalog['clean/y'])
    test_data = pd.read_csv(catalog['clean/test'])
    return train_data, y_train, test_data

def lr_prediction():
    train_data, y_train, test_data = get_model_data()
    le = LinearRegression()
    model = le.fit(train_data, y_train)
    scoring = ('r2', 'neg_mean_squared_error')
    cv_results = cross_validate(model,
                                train_data,
                                y_train,
                                cv=3,
                                scoring=scoring,
                                return_train_score=True)
    model_metrics = {metric: sorted(cv_results['test_'+metric], reverse=True)[0]
                     for metric in scoring}
    y_pred_array = model.predict(test_data)
    input_dict = {"Id": test_data['Id'].to_numpy(), 'SalePrice': y_pred_array.flatten()}
    y_pred_df = pd.DataFrame(input_dict)
    y_pred_df.to_csv(catalog['business/lr_predictions'], index=False)
    return y_pred_df, model_metrics

def ridge_prediction():
    train_data, y_train, test_data = get_model_data()
    ridge = Ridge()
    hyperparameters = {
        "alpha": range(0, 100, 5)
    }
    scoring = ('r2', 'neg_mean_squared_error')
    grid_model = GridSearchCV(ridge, hyperparameters,
                              scoring=scoring,
                              cv=3,
                              refit='r2'
                              )
    grid_model.fit(train_data, y_train)
    model = grid_model.best_estimator_
    model_metrics = {metric: grid_model.cv_results_['mean_test_'+metric][grid_model.best_index_] for metric in scoring}
    y_pred_array = model.predict(test_data)
    input_dict = {"Id": test_data['Id'].to_numpy(), 'SalePrice': y_pred_array.flatten()}
    y_pred_df = pd.DataFrame(input_dict)
    y_pred_df.to_csv(catalog['business/ridge_predictions'], index=False)
    return y_pred_df, model_metrics

def lasso_prediction():
    train_data, y_train, test_data = get_model_data()
    lasso = Lasso()
    hyperparameters = {
        "alpha": range(50, 100, 5),
        "selection": ['random'],
        "warm_start": [True],
        "tol": np.arange(1e-4, 2e-4, 1e-5),
        "max_iter": [1000]
    }
    scoring = ('r2', 'neg_mean_squared_error')
    grid_model = GridSearchCV(lasso, hyperparameters,
                              scoring=scoring,
                              cv=3,
                              refit='r2',
                              verbose=0
                              )
    grid_model.fit(train_data, y_train)
    model = grid_model.best_estimator_
    model_metrics = {metric: grid_model.cv_results_['mean_test_'+metric][grid_model.best_index_] for metric in scoring}
    # model_params = grid_model.best_params_
    # d = {feature: coef for feature, coef in zip(model.feature_names_in_, model.coef_) if coef != 0}
    y_pred_array = model.predict(test_data)
    input_dict = {"Id": test_data['Id'].to_numpy(), 'SalePrice': y_pred_array.flatten()}
    y_pred_df = pd.DataFrame(input_dict)
    y_pred_df.to_csv(catalog['business/lasso_predictions'], index=False)
    return y_pred_df, model_metrics

def elastinet_prediction():
    train_data, y_train, test_data = get_model_data()
    enet = ElasticNet()
    hyperparameters = {
        "alpha": range(0, 50, 5),
        "l1_ratio": np.arange(0, 1, 0.2),
        "selection": ['random'],
        "warm_start": [False],
        # "tol": np.arange(1.2e-4),
        "max_iter": [1000]
    }
    scoring = ('r2', 'neg_mean_squared_error')
    grid_model = GridSearchCV(enet, hyperparameters,
                              scoring=scoring,
                              cv=3,
                              refit='r2',
                              verbose=0
                              )
    grid_model.fit(train_data, y_train)
    model = grid_model.best_estimator_
    model_metrics = {metric: grid_model.cv_results_['mean_test_'+metric][grid_model.best_index_] for metric in scoring}
    # model_params = grid_model.best_params_
    # d = {feature: coef for feature, coef in zip(model.feature_names_in_, model.coef_) if coef != 0}
    y_pred_array = model.predict(test_data)
    input_dict = {"Id": test_data['Id'].to_numpy(), 'SalePrice': y_pred_array.flatten()}
    y_pred_df = pd.DataFrame(input_dict)
    y_pred_df.to_csv(catalog['business/enet_predictions'], index=False)
    return y_pred_df, model_metrics

def deeplearning_prediction():
    train_data, y_train, test_data = get_model_data()

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_shape=(212,), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    model = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
    scoring = ('r2', 'neg_mean_squared_error')
    cv_results = cross_validate(model,
                                train_data,
                                y_train,
                                cv=3,
                                scoring=scoring,
                                return_train_score=True)
    model_metrics = {metric: sorted(cv_results['test_'+metric], reverse=True)[0]
                     for metric in scoring}
    y_pred_array = model.predict(test_data)
    input_dict = {"Id": test_data['Id'].to_numpy(), 'SalePrice': y_pred_array.flatten()}
    y_pred_df = pd.DataFrame(input_dict)
    y_pred_df.to_csv(catalog['business/dlearn_predictions'], index=False)
    return y_pred_df, model_metrics


if __name__ == "__main__":
    main()