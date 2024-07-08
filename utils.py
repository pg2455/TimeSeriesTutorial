import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils_tfb

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

plt.style.use('tableau-colorblind10')
PLOTTING_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def load_tutotrial_data(dataset):
    """Loads dataset for tutorial."""
    TS_DATA_FOLDER = pathlib.Path("./forecasting").resolve()
    if dataset == 'exchange_rate':
        dataset = TS_DATA_FOLDER / "Exchange.csv"
        data = utils_tfb.read_data(str(dataset))
        data.index.freq = 'D'  # since we know that the frequency is daily
        data = data.resample("W").mean()
        return data
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")


def get_cols(time_series_data, cols):
    if cols is None:
        cols = time_series_data.columns

    if len(cols) > 10:
        return random.sample(cols, 10)
    
    return cols


def plot_raw_data(time_series_data, cols=None, figsize=(15, 10)):
    """Plots the time series in data. If there are more than 10, it samples `n` of them.
    
    Args:
        time_series_data: pandas dataframe containing time series in each column and index as a PeriodIndex
        cols: list of columns names in time_series_data to plot
    """
    cols = get_cols(time_series_data, cols)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    for idx, col in enumerate(cols):
        ax.plot(time_series_data[col], label=f'{col}')
    
    ax.grid()
    ax.legend()
    fig.suptitle("Raw data")
    return fig, ax


def plot_seasonality_decompose(time_series_data, cols=None, figsize=(15, 15)):
    """Seasonality decomposition for the time series. 
    Args:
        time_series_data: pandas dataframe containing time series in each column and index as a PeriodIndex
        cols: list of columns names in time_series_data to plot
    """
    
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize, dpi=100)
    cols = get_cols(time_series_data, cols)
    
    for idx, col in enumerate(cols):
        ts = time_series_data[col]
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        
        # original series
        ax = axs[0]
        ax.plot(ts.values, color=color, label=f'{col}')
        ax.set_title('Original')
        ax.set_xticks([])
        
        # seasonal decomposition
        result = seasonal_decompose(ts, model="additive")
        
        # trend
        ax = axs[1]
        ax.plot(result.trend, color=color, label=f'{col}')
        ax.set_title('Trend')
        ax.set_xticks([])
        
        # seasonality
        ax = axs[2]
        ax.plot(result.seasonal, color=color, label=f'{col}: {result.seasonal.mean(): 0.4f}', alpha=1/(idx+1))
        ax.set_title('Seasonality')
        ax.set_xticks([])
        
        # residual
        ax = axs[3]
        ax.plot(result.resid, color=color, label=f'{col}: {result.resid.mean(): 0.4f}', alpha=1/(idx+1))
        ax.set_title('Residuals')
        
        for ax in axs:
            ax.legend()
            ax.grid()

    fig.suptitle("Seasonality decomposition")
    return fig, axs


def plot_acf_pacf(time_series_data, cols=None, figsize=(10, 10)):
    """Plots autocorrelation and partial autocorrelations."""
    lags = np.arange(150)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=100)
    cols = get_cols(time_series_data, cols)

    ax = axs[0]
    for idx, col in enumerate(cols):
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        ts = time_series_data[col]
        autocorrs = [ts.autocorr(lag=lag) for lag in lags]
        ax.plot(lags, autocorrs, label=f'{col}', color=color)

    ax.set_xlim(0, max(lags))
    ax.hlines(0, 0, max(lags), color='red', linewidth=3, linestyle='--')
    ax.legend()
    ax.grid()
    ax.set_title("Autocorrelations")

    ax = axs[1]
    for idx, col in enumerate(cols):
        color = PLOTTING_COLORS[idx % len(PLOTTING_COLORS)]
        ts = time_series_data[col]
        _ = plot_pacf(ts, method='ywm', lags=lags, ax=ax, title=None, label=f'{col}')

    ax.set_title("Partial Autocorrelations")
    return fig, axs


def check_stationarity(time_series_data, cols=None):
    """Checks whether the time series is stationary using ADF test."""
    cols = get_cols(time_series_data, cols)
    alpha = 0.05

    for col in cols:
        ts = time_series_data[col]
        results = adfuller(ts)
        pvalue = results[1]
        print(f"Col: {col}\tP-value: {pvalue:0.4f}\tStationary: {pvalue < alpha}")


def mean_absolute_error(forecast, target):
    """Computes Mean Absolute Error."""
    return np.mean(np.abs(forecast - target))


def mean_squared_error(forecast, target):
    """Computes the mean squared error."""
    return np.mean(np.power(forecast - target, 2))


def root_relative_squared_error(forecast, target, train):
    """Returns the scaled MSE, where the scale is mean on the training data as forecast"""
    mse = mean_squared_error(forecast, target)
    mean = np.mean(train)
    naive_mse = mean_squared_error(target, mean)
    return mse / naive_mse


def mean_absolute_scaled_error(forecast, target, insample_data, seasonality=1):
    """
    Computes Mean Absolute Scaled Error. 
    
    Args:
        target (list, np.array, pd.Series): ground truth time series
        forecast (list, np.array, pd.Series): forecasted values for target
        insample_data (list, np.array, pd.Series): training data for forecast
    """
    insample_naive_forecast = insample_data[:-seasonality] # our naive predictions
    insample_targets = insample_data[seasonality:] # our targets are the values after the shift by seasonality
    #
    mae = mean_absolute_error(forecast, target)
    mae_naive = mean_absolute_error(insample_naive_forecast, insample_targets)
    #
    eps = np.finfo(np.float64).eps
    #
    mase = mae / np.maximum(mae_naive, eps)
    return mase


def get_all_metrics(forecast, target, insample_data, seasonality=1):
    mse = mean_squared_error(forecast, target)
    rse = root_relative_squared_error(forecast, target, insample_data)
    mae = mean_absolute_error(forecast, target)
    mase = mean_absolute_scaled_error(forecast, target, insample_data, seasonality)
    return {'mae':mae, 'mase':mase, 'rse': rse, 'mse':mse}


def get_mase(forecasted_df, target_df, train_df, horizon=-1):
    """Multivariate Time Series: Computes mase for each column.

    Args:
        forecasted_df (pd.DataFrame): forecasts of the same shape as target_df
        target_df (pd.DataFrame): target for forecasts
        train_df (pd.DataFrame): historical data for training
        horizon (int): horizon over which to compute MASE
    """
    if horizon < 0:
        horizon = forecasted_df.shape[0]

    all_mase = {}
    for col in train_df.columns:
        mase = mean_absolute_scaled_error(
            forecasted_df[col].values[:horizon], 
            target_df[col].values[:horizon], 
            train_df[col].values
        )
        all_mase[col] = mase

    return all_mase

def highlight_min(data, color='white', font_weight='bold'):
    """Returns the style for each cell."""
    attr = f'background-color: {color}; font-weight: {font_weight}'
    if data.ndim == 1:  # Apply column-wise
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # Apply element-wise if needed
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''), index=data.index, columns=data.columns)


def plot_forecasted_series(model_results, train_data, test_data, n_history_to_plot=100, forecasting_horizon=100, col=None):
    """Plots forecasted series in continuation with the historical data.
    Args:
        model_results (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): trained model results.
        train_data (pd.DataFrame): pandas dataframe containing time series in each column and index as a PeriodIndex
        test_data (pd.DataFrame): pandas dataframe containing time series for targets in each column and index as a PeriodIndex
        n_hisrtory_to_plot (int): number of points from train_data to plot
        forecasting_horizon (int): number of points to project forward in time
        col (str): a valid column in train_data
    
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    color = PLOTTING_COLORS[0]
    
    train_data[col][-n_history_to_plot:].plot(ax=ax, color=color, label='train series')
    
    # forecasts
    predictions = model_results.get_forecast(steps=forecasting_horizon)
    mean = predictions.predicted_mean
    se = predictions.se_mean
    
    #
    mean.plot(ax=ax, color=color, linestyle=':', linewidth=2, label='estimated mean')
    ax.fill_between(se.index, mean.values - se.values, se.values + mean.values, color=color, alpha=0.4)
    test_data[col][:forecasting_horizon].plot(ax=ax, color=color, linestyle='--', label='target')
    (mean-se).plot(ax=ax, color='red', linestyle='--')
    (mean+se).plot(ax=ax, color='red', linestyle='--')
    
    ax.legend()

    fig.suptitle(f"Column: {col}")
    return fig, ax


def update_results(records, return_original_if_assert_false=True):
    """Updates results with records. Ensures that the models are not repeated."""
    results = pd.read_csv('metrics_tutorial_exchange_rate.csv')
    print(f"Existing size of the results: {results.shape}")
    print(f"Existing methods in results: {results.model.unique()}")

    new_methods = set([x['model'] for x in records])

    if sum(x in results.model.unique() for x in new_methods) > 0:
        print( "The model already exist in the results. Returning original.")
        return results

    # Store these metrics to be able to compare them against other methods later on
    more_results = pd.DataFrame(records)
    results = pd.concat([results, more_results])
    results.to_csv('metrics_tutorial_exchange_rate.csv', index=False)

    print(f"New size of the results: {results.shape}")
    print(f"New methods in results: {results.model.unique()}")

    return results


def display_results(results):
    """Displays results so that the minimum is highlighted across different horizons."""
    mean_df = results.groupby(['horizon', 'model']).mean(['mase', 'mae', 'rse', 'mse']).reset_index()
    mean_df['col'] = 'overall'

    results = pd.concat([results, mean_df])

    results_df = results.pivot_table(index=['model'], columns=['horizon', 'col'], values='mase')
    for h in results.horizon.unique():
        print("#"*50, f"Horizon: {h}", "#"*50)
        df = results_df[h]
        styled_df = df.style.apply(highlight_min, axis=0)
        display(styled_df)
        print("\n")


def update_test_predictions(test_predictions, return_original_if_assert_false=True):
    """Updates test predictions in the CSV. """

    prev_test_predictions = pd.read_csv("test_predictions_tutorial_exchange_rate.csv", index_col='date', parse_dates=['date'])

    more_test_predictions = pd.DataFrame(test_predictions)
    more_test_predictions.index = prev_test_predictions.index

    if sum(x in prev_test_predictions.columns for x in more_test_predictions.columns) > 0:
        print("Columns already exists. Returning the existing test predictions.")
        return prev_test_predictions

    test_predictions = pd.concat([prev_test_predictions, more_test_predictions], axis=1)
    print("New columns:", test_predictions.columns)

    test_predictions.to_csv("test_predictions_tutorial_exchange_rate.csv", index=True)
    return test_predictions
