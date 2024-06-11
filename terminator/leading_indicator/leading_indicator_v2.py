import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import beta
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
from snowflake import connector
import warnings
warnings.filterwarnings('ignore', category=Warning)

from numpy import asarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from functools import partial

from snowflake.connector.pandas_tools import write_pandas
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

################################################
################ Ingest & Clean ##############
################################################


#Create a function to ingest data and clean it 
def run_queries(connection, start_date = '2024-04-01'):
    
    #SQL Queries:
    raw_daily_display_query = f"""
    select 
        to_date(event_created_at) as event_date,
        -- widget_type,
        -- ml_model,
        case 
            when widget_type = 'DISCOFEED' and ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
            when ml_model ilike '%waterfall%' then 'waterfall'
            when widget_type = 'LEAD_GEN' then 'nurture'
            when ml_model = 'contextual' and widget_type != 'LEAD_GEN' then 'contextual_classic'
            else 'other' 
        end as load_type,
        count(distinct event_id) as brand_displays
    from event.event e 
    where
        event_name = 'widget_brand_display'
        and to_date(event_created_at) between '{start_date}' and current_date -1
        -- and ml_model is null
        and brand_name != publisher_name
    group by all 
    order by event_date asc
    """

    raw_daily_display_click_query = f"""
    with displays as (
    select 
        to_date(event_created_at) as event_date,
        -- widget_type,
        -- ml_model,
        case 
            when widget_type = 'DISCOFEED' and ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
            when ml_model ilike '%waterfall%' then 'waterfall'
            when widget_type = 'LEAD_GEN' then 'nurture'
            when ml_model = 'contextual' and widget_type != 'LEAD_GEN' then 'contextual_classic'
            else 'other' 
        end as load_type,
        count(distinct event_id) as brand_displays
    from event.event e 
    where
        event_name = 'widget_brand_display'
        and to_date(event_created_at) between '{start_date}' and current_date -1
        -- and ml_model is null
        and brand_name != publisher_name
    group by all),
    clicks as (
    select 
        to_date(event_created_at) as event_date,
        -- widget_type,
        -- ml_model,
        case 
            when widget_type = 'DISCOFEED' and ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
            when ml_model ilike '%waterfall%' then 'waterfall'
            when widget_type = 'LEAD_GEN' then 'nurture'
            when ml_model = 'contextual' and widget_type != 'LEAD_GEN' then 'contextual_classic'
            else 'other' 
        end as load_type,
        count(distinct event_id) as clicks
    from event.event e 
    where
        event_name in ('widget_brand_click', 'widget_product_click')
        and to_date(event_created_at) between '{start_date}' and current_date -1
        -- and ml_model is null
        and brand_name != publisher_name
    group by all 
    )
    select 
        event_date,
        load_type,
        brand_displays,
        case when clicks is null then 0 else clicks end as clicks
    from displays d 
    left join clicks c using (event_date, load_type)
    order by event_date asc
    ;
    """

    raw_zero_day_adspend_query = f"""
    select 
        to_date(event_created_at) as event_date,
        case 
            when e.widget_type = 'DISCOFEED' and e.ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
            when e.ml_model ilike '%waterfall%' then 'waterfall'
            when e.widget_type = 'LEAD_GEN' then 'nurture'
            when e.ml_model = 'contextual' and e.widget_type != 'LEAD_GEN' then 'contextual_classic'
            else 'other' 
        end as load_type,
        sum(case when billable_amount is null then 0 else billable_amount end) as zeroday_ad_spend
    from curated.ad_spend_revenue asr 
    left join event.event e using (event_id)
    where 
        conversion_type = 'cross-sell'
        and to_date(event_created_at) between '{start_date}' and current_date -1
        and days_to_attribution = 0
    group by all 
    order by event_date asc
    ;
    """

    final_conv_query = f"""
    select 
        to_date(c.event_created_at) as event_date,
        count(distinct c.order_id) as final_conv_count,
        sum(case when billable_amount is null then 0 else billable_amount end) as final_ad_spend,
    from curated.ad_spend_revenue c
    where 
        c.conversion_type = 'cross-sell'
        and to_date(c.event_created_at) between '{start_date}' and current_date -1
        -- and c.days_to_attribution = 0
    group by all 
    order by event_date asc
    ;
    """

    daily_order_query = f"""
    select 
        to_date(created_at_gmt) as event_date,
        count(distinct order_id) as order_count
    from orders.order_data_flattened
    where 
        to_date(created_at_gmt) between '{start_date}' and current_date-1
    group by all
    order by event_date asc
    ;
    """

    annual_seasonality_query = f"""
    with loads as (
    select 
        date,
        sum(publisher_widget_loads) as loads
    from curated.combined_discofeed_daily
    where 
        date between '2023-01-01' and '2023-12-31'
    group by all),
    revenue as (
    select 
        to_date(conversion_order_time) as date,
        count(distinct conversion_unique_key) as conv_count,
        sum(bod_spend) as ad_spend,
    from curated.legacy_ad_spend_revenue
    where 
        date between '2023-01-01' and '2023-12-31'
    group by all 
    ),
    daily as (
    select 
        *,
        conv_count / loads as cvr,
        ad_spend / loads as rpl
    from loads 
    join revenue using (date)
    order by date asc),
    averages as (
    select 
        avg(loads) as avg_loads,
        avg(conv_count) as avg_conv,
        avg(ad_spend) as avg_adspend,
        avg(cvr) as avg_cvr,
        avg(rpl) as avg_rpl,
    from daily 
    )
    select 
        date as event_date, 
        loads / (select avg_loads from averages) as loads_index,
        conv_count / (select avg_conv from averages) as conv_index,
        ad_spend / (select avg_adspend from averages) as adspend_index,
        cvr / (select avg_cvr from averages) as cvr_index,
        rpl / (select avg_rpl from averages) as rpl_index,
    from daily
    ;
    """
    #Run Queries
    raw_daily_display = pd.read_sql(raw_daily_display_query, connection)
    raw_daily_display_click = pd.read_sql(raw_daily_display_click_query, connection)
    raw_zero_day_adspend = pd.read_sql(raw_zero_day_adspend_query, connection)
    final_conv = pd.read_sql(final_conv_query, connection)
    daily_order = pd.read_sql(daily_order_query, connection)
    annual_seasonality = pd.read_sql(annual_seasonality_query, connection)

    return raw_daily_display, raw_daily_display_click, raw_zero_day_adspend, final_conv, daily_order, annual_seasonality

def pivot_data(df, value, index = 'event_date', column = 'load_type'):
    #lowercase the columns
    df.columns = df.columns.str.lower()
    #Pivot the data by load type
    pivot = df.pivot(index=index, columns=column, values=value).reset_index()
    #fill NaN values with 0
    pivot = pivot.fillna(0)
    #remove the index name
    pivot.columns.name = None
    #add a subscript of "_displays" to all of the columns except 'event_data' 
    pivot.columns = ['event_date'] + [col + '_' + value for col in pivot.columns[1:]]
    return pivot
    
def merge_data(raw_daily_display, raw_daily_display_click, raw_zero_day_adspend, final_conv, daily_order, annual_seasonality):
    #Pivot the raw daily data
    display_pivot = pivot_data(raw_daily_display, 'brand_displays')
    zero_day_adspend_pivot = pivot_data(raw_zero_day_adspend, 'zeroday_ad_spend')

    #Merge the two pivoted datasets
    merged_df = pd.merge(display_pivot, zero_day_adspend_pivot, on='event_date')
    
    #Lowercase the columns for the daily order, final conversion, and annual seasonality data
    daily_order.columns = daily_order.columns.str.lower()
    final_conv.columns = final_conv.columns.str.lower()
    annual_seasonality.columns = annual_seasonality.columns.str.lower()

    #merge the daily order data
    merged_df = pd.merge(merged_df, daily_order, on='event_date')
    #merge the final conversion data
    merged_df = pd.merge(merged_df, final_conv, on='event_date')
    #calculate rpl as final_ad_spend divided by the sum of all the display columns
    merged_df['rpl'] = round(merged_df['final_ad_spend'] / merged_df.iloc[:, 1:5].sum(axis=1),3)
    #convert the event_date column to datetime
    merged_df['event_date'] = pd.to_datetime(merged_df['event_date'])
    #Add a day of week column
    merged_df['day_of_week'] = merged_df['event_date'].dt.dayofweek
    #Add a day count column that counts the number of days since the start of the dataset
    # merged_df['day_count'] = range(1, len(merged_df)+1)
    #Add a day of month column
    merged_df['day_of_month'] = merged_df['event_date'].dt.day

    #Add Seasonality via inner join of annual seasonality's date column
    annual_seasonality = annual_seasonality[['event_date', 'rpl_index']]
    #convert the event_date column to datetime for annual seasonality
    annual_seasonality['event_date'] = pd.to_datetime(annual_seasonality['event_date'])
    #Add one year to the annual seasonality date column
    annual_seasonality['event_date'] = annual_seasonality['event_date'] + pd.DateOffset(years=1)
    merged_df = pd.merge(merged_df, annual_seasonality, on='event_date', how='left')
    return merged_df

def ingest_clean_data(connection, start_date = '2024-04-01'):
    #Print a status message
    print(f'Ingesting performance data from {start_date}')
    raw_daily_display, raw_daily_display_click, raw_zero_day_adspend, final_conv, daily_order, annual_seasonality = run_queries(connection)
    print('Data successfully ingested.')
    merged_df = merge_data(raw_daily_display, raw_daily_display_click, raw_zero_day_adspend, final_conv, daily_order, annual_seasonality)
    return merged_df

################################################
################ Model Training ################
################################################

def scale_data(df):
    scaler = StandardScaler()
    #copy the merged_df dataframe
    scaled_df = df.copy()

    #isolate the features as all columns except 'event_date', 'rpl', 'conversion_rate', 'final_conv_count', 'final_ad_spend' 
    X = scaled_df.drop(columns=['event_date', 'rpl', 'final_conv_count', 'final_ad_spend', 'day_of_week', 'day_of_month', 'rpl_index'])
    y = df[['rpl', 'final_ad_spend']]

    # #scale the input data
    X_scaled = scaler.fit_transform(X)
    #Append the scaled data to the scaled_df dataframe
    scaled_df[X.columns] = X_scaled
    return pd.DataFrame(scaled_df)

def remove_last_n_days(df, n):
    #Cut off the last 14 days of data
    current_date = datetime.today().date()
    #set cutoff date to 14 days before the current date
    cutoff_date = pd.Timestamp(current_date - pd.Timedelta(days=n))
    #filter out all rows where the event_date is greater than the cutoff date
    unattributed_days = df[df['event_date'] >= cutoff_date]
    attributed_days = df[df['event_date'] < cutoff_date]
    #Print the shape of the two datasets
    print(f'Total Days: {df.shape}')
    print(f'Unattributed Days: {unattributed_days.shape}')
    print(f'Attributed Days: {attributed_days.shape}')
    return unattributed_days, attributed_days

def optimize_random_forest_hyperparameters(X_train, y_train):

    # Define the model
    rf = RandomForestRegressor(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Define the scorer
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup the grid search
    grid_search = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            scoring=scorer,
                            cv=5,  # 5-fold cross-validation
                            n_jobs=-1,  # Use all available cores
                            verbose=1)

    # Fit grid search to the train set
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters:", best_params)
    return best_model

def train_test_split_latest_data(X, y, training_size):
    #Determine the nearest integer to % of the dataset to use for training
    train_size = round(len(X) * training_size)
    #Set training data 
    X_train = X[:train_size]
    y_train = y[:train_size]
    #Set test data
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test

def find_model_performance(scaled_df, objective = 'rpl', random_sample = True, training_size = 0.8):
    X = scaled_df.copy()
    X.drop(columns=['event_date', 'final_conv_count', 'final_ad_spend', 'rpl'], inplace=True)
    y = scaled_df[objective]
    
    if random_sample != True:
        X_train, X_test, y_train, y_test = train_test_split_latest_data(X, y, training_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=42)

    # Fit the Linear Regression model
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_lr = linear_reg.predict(X_test)
    y_train_pred_lr = linear_reg.predict(X_train)  # Predictions on the training set

    # Calculate Errors on test and train data for Linear Regression
    lr_rmse_test = mean_squared_error(y_test, y_pred_lr, squared=False)
    lr_rmse_train = mean_squared_error(y_train, y_train_pred_lr, squared=False)
    mean_abs_error_pct_lr_test = round((abs(y_test - y_pred_lr) / y_test).mean(),3)
    mean_abs_error_pct_lr_train = round((abs(y_train - y_train_pred_lr) / y_train).mean(),3)

    #Optimize the Random Forest model and predict on the test data
    best_rf_model = optimize_random_forest_hyperparameters(X_train, y_train)
    y_pred_rf = best_rf_model.predict(X_test)
    y_train_pred_rf = best_rf_model.predict(X_train)  # Predictions on the training set

    # Calculate Errors on test and train data for Random Forest
    rf_rmse_test = mean_squared_error(y_test, y_pred_rf, squared=False)
    rf_rmse_train = mean_squared_error(y_train, y_train_pred_rf, squared=False)
    mean_abs_error_pct_rf_test = round((abs(y_test - y_pred_rf) / y_test).mean(),3)
    mean_abs_error_pct_rf_train = round((abs(y_train - y_train_pred_rf) / y_train).mean(),3)

    # Save the results to a dataframe with additional columns for training errors
    results = pd.DataFrame({
        'model': ['Linear Regression', 'Random Forest'],
        'RMSE Test': [lr_rmse_test, rf_rmse_test],
        'RMSE Train': [lr_rmse_train, rf_rmse_train],
        'Mean Abs Error % Test': [mean_abs_error_pct_lr_test, mean_abs_error_pct_rf_test],
        'Mean Abs Error % Train': [mean_abs_error_pct_lr_train, mean_abs_error_pct_rf_train]
    })

    #Plot a scatter plot of the actual vs predicted values for both models with different colors for each model and a y=x line
    plt.figure(figsize=(8,3))
    plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
    plt.scatter(y_test, y_pred_rf, color='red', label='Random Forest')
    #Insert label based on the objective variable
    plt.xlabel(f'Actual {objective}')
    plt.ylabel(f'Predicted {objective}')
    plt.title(f'Actual vs Predicted {objective} by Model Type')
    plt.legend()    
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black')
    plt.show()

    return results, linear_reg, best_rf_model

def perform_cv(validation_data , params, param_grid, num_rounds=1000, nfolds=5, early_stopping_rounds=50):
        cv_results = {}
        min_mae = float("Inf")
        best_params = None

        for eta in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample in param_grid['colsample_bytree']:
                        for lam in param_grid['lambda']:
                            for alpha in param_grid['alpha']:
                                # print(f"CV with learning_rate={eta}, max_depth={max_depth}, subsample={subsample}, colsample={colsample}, lambda={lam}, alpha={alpha}")
                                # Update our parameters
                                params['eta'] = eta
                                params['max_depth'] = max_depth
                                params['subsample'] = subsample
                                params['colsample_bytree'] = colsample
                                params['lambda'] = lam
                                params['alpha'] = alpha
                                
                                # Run CV
                                cv_result = xgb.cv(
                                    params,
                                    validation_data,
                                    num_boost_round=num_rounds,
                                    seed=42,
                                    nfold=nfolds,
                                    metrics={'mae'},
                                    early_stopping_rounds=early_stopping_rounds
                                )

                                # Update best MAE
                                mean_mae = cv_result['test-mae-mean'].min()
                                boost_rounds = cv_result['test-mae-mean'].argmin()
                                # print(f"MAE {mean_mae} for {boost_rounds} rounds")
                                if mean_mae < min_mae:
                                    min_mae = mean_mae
                                    best_params = (eta, max_depth, subsample, colsample, lam, alpha)
                                    # print(f"New best params: {best_params}, MAE: {min_mae}")

        print(f"Best params: {best_params}, Minimum MAE: {min_mae}")
        return best_params

def tune_xgboost_parameters(validation_data, params):
    # Define the parameter grid
    param_grid = {
        'learning_rate': [1, 0.01, 0.001],
        'max_depth': [5, 4,3],
        'subsample': [0.8, 0.6],
        'colsample_bytree': [0.8, 0.6],
        'lambda': [1, 0.1, 0.001],  # L2 regularization
        'alpha': [1, 0.1, 0.001]    # L1 regularization
    }

    best_params = perform_cv(validation_data, params, param_grid)
    return best_params

def train_and_optimize_xg_model(scaled_df, training_size = 0.7, validation_size = 0.1, objective = 'rpl', random_sample = True):
    X = scaled_df.copy()
    X.drop(columns=['event_date', 'final_conv_count', 'final_ad_spend', 'rpl'], inplace=True)
    y = scaled_df[objective]


    if random_sample != True:
        X_train, X_rest, y_train, y_rest = train_test_split_latest_data(X, y, training_size)
        X_val, X_test, y_val, y_test = train_test_split_latest_data(X_rest, y_rest, training_size=round(validation_size / (1 - training_size)))
    else:
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=training_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=validation_size / (1 - training_size), random_state=42)

    #Initialize the XGBoost model
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)
    valid_data = xgb.DMatrix(X_val, label=y_val)

    # Set the parameters for the XGBoost model
    initial_params = {
        'learning_rate': 0.001,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        # 'lambda': 1,  # L2 regularization
        # 'alpha': 0.1  # L1 regularization  
    }

    def mape_objective(preds, dtrain):
        labels = dtrain.get_label()
        # Avoid division by zero
        elements = np.where(labels != 0, np.abs((labels - preds) / labels), 0)
        grad = np.where(labels != 0, -100 * (labels - preds) / (labels ** 2), 0)  # Gradient
        hess = np.where(labels != 0, 100 / np.abs(labels), 0)  # Hessian

        return grad, hess

    #Train the data
    num_rounds = 1000
    watchlist = [(train_data, 'train'), (valid_data, 'valid')]

    initial_model = xgb.train(
        initial_params,
        train_data,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        obj=mape_objective,  # Use the custom MAPE objective function
        feval=None,  # Temporarily set feval to None
        maximize=False,
        verbose_eval=200,
        )

    # Evaluate the performance of the model on the test set
    y_pred_xgb_initial = initial_model.predict(test_data)
    y_train_pred_xgb_initial = initial_model.predict(train_data)

    # Calculate Errors on test and train data for XGBoost
    # xgb_rmse_test_initial = mean_squared_error(y_test, y_pred_xgb_initial, squared=False)
    # xgb_rmse_train_initial = mean_squared_error(y_train, y_train_pred_xgb_initial, squared=False)
    mean_abs_error_pct_xgb_test_initial = round((abs(y_test / y_pred_xgb_initial) - 1).mean(),3)
    mean_abs_error_pct_xgb_train_initial = round((abs(y_train / y_train_pred_xgb_initial) - 1).mean(),3)

    #Print Status Message
    print('Initialize Model Training')
    # Save the results to a dataframe with additional columns for training errors
    initial_results = pd.DataFrame({
        'model': ['XGBoost_initial'],
        # 'RMSE Test': [xgb_rmse_test_initial],
        # 'RMSE Train': [xgb_rmse_train_initial],
        'Mean Abs Error % Test': [mean_abs_error_pct_xgb_test_initial],
        'Mean Abs Error % Train': [mean_abs_error_pct_xgb_train_initial]
    })

    # Tune the XGBoost model
    print('Tuning Model')
    best_params = tune_xgboost_parameters(valid_data, initial_params)

    # Update the parameters with the best parameters
    optimized_params = {
        'learning_rate': best_params[0],
        'max_depth': best_params[1],
        'subsample': best_params[2],
        'colsample_bytree': best_params[3],
        # 'objective': 'reg:squarederror',
        'lambda': best_params[4],  # L2 regularization
        'alpha': best_params[5]  # L1 regularization  
    }

    # Train the model with the best parameters
    best_model = xgb.train(
        optimized_params,
        train_data,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        obj=mape_objective,  # Use the custom MAPE objective function
        feval=None,  # Temporarily set feval to None
        maximize=False,
        verbose_eval=200,
    )

    # Evaluate the performance of the model on the test set
    y_pred_xgb_optimized = best_model.predict(test_data)
    y_train_pred_xgb_optimized = best_model.predict(train_data)

    # Calculate Errors on test and train data for XGBoost
    # xgb_rmse_test_optimized = mean_squared_error(y_test, y_pred_xgb_optimized, squared=False)
    # xgb_rmse_train_optimized = mean_squared_error(y_train, y_train_pred_xgb_optimized, squared=False)
    mean_abs_error_pct_xgb_test_optimized = round((abs(y_test - y_pred_xgb_optimized) / y_test).mean(),3)
    mean_abs_error_pct_xgb_train_optimized = round((abs(y_train - y_train_pred_xgb_optimized) / y_train).mean(),3)

    # Save the results to a dataframe with additional columns for training errors
    results = pd.DataFrame({
        'model': ['XGBoost_best'],
        # 'RMSE Test': [xgb_rmse_test_optimized],
        # 'RMSE Train': [xgb_rmse_train_optimized],
        'Mean Abs Error % Test': [mean_abs_error_pct_xgb_test_optimized],
        'Mean Abs Error % Train': [mean_abs_error_pct_xgb_train_optimized]
    })

    #union the two dataframes
    results = pd.concat([initial_results, results])

    return results, initial_model, best_model

def train_xg_model(scaled_df, training_size = 0.7, objective = 'rpl', random_sample = True):
    X = scaled_df.copy()
    X.drop(columns=['event_date', 'final_conv_count', 'final_ad_spend', 'rpl'], inplace=True)
    y = scaled_df[objective]

    if random_sample != True:
        X_train, X_test, y_train, y_test = train_test_split_latest_data(X, y, training_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=42)

    #Initialize the XGBoost model
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    # Set the parameters for the XGBoost model
    initial_params = {
        'learning_rate': 0.001,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        # 'lambda': 1,  # L2 regularization
        # 'alpha': 0.1  # L1 regularization  
    }

    def mape_objective(preds, dtrain):
        labels = dtrain.get_label()
        # Avoid division by zero
        elements = np.where(labels != 0, np.abs((labels - preds) / labels), 0)
        grad = np.where(labels != 0, -100 * (labels - preds) / (labels ** 2), 0)  # Gradient
        hess = np.where(labels != 0, 100 / np.abs(labels), 0)  # Hessian

        return grad, hess

    #Train the data
    num_rounds = 1000
    watchlist = [(train_data, 'train')]

    initial_model = xgb.train(
        initial_params,
        train_data,
        num_boost_round=1000,
        evals=watchlist,
        early_stopping_rounds=50,
        obj=mape_objective,  # Use the custom MAPE objective function
        feval=None,  # Temporarily set feval to None
        maximize=False,
        verbose_eval=200,
        )

    # Evaluate the performance of the model on the test set
    y_pred_xgb_initial = initial_model.predict(test_data)
    y_train_pred_xgb_initial = initial_model.predict(train_data)

    # Calculate Errors on test and train data for XGBoost
    xgb_rmse_test_initial = mean_squared_error(y_test, y_pred_xgb_initial, squared=False)
    xgb_rmse_train_initial = mean_squared_error(y_train, y_train_pred_xgb_initial, squared=False)
    mean_abs_error_pct_xgb_test_initial = round((abs(y_test / y_pred_xgb_initial) - 1).mean(),3)
    mean_abs_error_pct_xgb_train_initial = round((abs(y_train / y_train_pred_xgb_initial) - 1).mean(),3)

    # Save the results to a dataframe with additional columns for training errors
    initial_results = pd.DataFrame({
        'model': ['XGBoost_initial'],
        'RMSE Test': [xgb_rmse_test_initial],
        'RMSE Train': [xgb_rmse_train_initial],
        'Mean Abs Error % Test': [mean_abs_error_pct_xgb_test_initial],
        'Mean Abs Error % Train': [mean_abs_error_pct_xgb_train_initial]
    })

    #Visualize the results of the model's predictions vs the actual values in a scatter plot
    plt.figure(figsize=(8,3))
    plt.scatter(y_test, y_pred_xgb_initial, color='blue', label='XGBoost')
    #Insert label based on the objective variable
    plt.xlabel(f'Actual {objective}')
    plt.ylabel(f'Predicted {objective}')
    plt.title(f'Actual vs Predicted {objective} for XgBoost')
    plt.legend()
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black')
    plt.show()

    return initial_results, initial_model

################################################
############# Prediction and Output ############
################################################

def predict_unattributed_days(unattributed_days, lr_model, rf_model, xgb_model, objective = 'rpl'):
    X_unattributed = unattributed_days.copy()
    X_unattributed.drop(columns=['event_date', 'final_conv_count', 'final_ad_spend', 'rpl'], inplace=True)
    y_unattributed = unattributed_days[objective]
    current_date = datetime.today().date()

    #Setup prediction output df 
    prediction_df = unattributed_days[['event_date']]
    prediction_df['model_date'] = current_date
    prediction_df['model_date'] = pd.to_datetime(prediction_df['model_date'])

    #Predict RPM for each model
    prediction_df['rpm_lr'] = lr_model.predict(X_unattributed).round(4)
    prediction_df['rpm_rf'] = rf_model.predict(X_unattributed).round(4)
    prediction_df['rpm_xgb'] = xgb_model.predict(xgb.DMatrix(X_unattributed)).round(4)

    prediction_df.to_csv(f'data/output/prediction_df_{current_date}.csv', index=False)
    return prediction_df


def create_and_write_db_table(df, table_name, schema='PLAYGROUND_ANALYTICS'):
    connection = connector.connect(
        user=os.getenv("SNOWFLAKE_USERNAME"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database='DISCO_CORE',
        schema= schema  # Set the schema here
    )

    #upper case the table name and all of the column names of the df
    table_name = table_name.upper()
    df.columns = [col.upper() for col in df.columns]
    
    # First, ensure the table exists with the correct schema
    create_table_query = f"""
    CREATE OR REPLACE TABLE PLAYGROUND_ANALYTICS.LEADING_INDICATOR_PREDICTIONS_FINGERPRINTING (
        EVENT_DATE DATE,
        MODEL_DATE DATE,
        RPM_LR FLOAT,
        RPM_RF FLOAT,
        RPM_XG FLOAT
    );
    """

    # Execute the create table query
    cursor = connection.cursor()
    cursor.execute(create_table_query)
    cursor.close()

    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE']).dt.date
    df['MODEL_DATE'] = pd.to_datetime(df['MODEL_DATE']).dt.date

    # Optionally convert to string if required by your specific setup
    df['EVENT_DATE'] = df['EVENT_DATE'].astype(str)
    df['MODEL_DATE'] = df['MODEL_DATE'].astype(str)

    # Now, write the DataFrame to the table
    write_pandas(
        conn=connection,
        df=df,
        table_name= table_name,
        auto_create_table=True  
    )

def update_to_db_table(df, table_name, schema='PLAYGROUND_ANALYTICS'):
    connection = connector.connect(
        user=os.getenv("SNOWFLAKE_USERNAME"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database='DISCO_CORE',
        schema= schema  # Set the schema here
    )

    #upper case the table name and all of the column names of the df
    table_name = table_name.upper()
    df.columns = [col.upper() for col in df.columns]
    
    #Modify date columns to be in the correct format
    df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE']).dt.date
    df['MODEL_DATE'] = pd.to_datetime(df['MODEL_DATE']).dt.date
    # Optionally convert to string if required by your specific setup
    df['EVENT_DATE'] = df['EVENT_DATE'].astype(str)
    df['MODEL_DATE'] = df['MODEL_DATE'].astype(str)

    # Now, write the DataFrame to the table
    write_pandas(
        conn=connection,
        df=df,
        table_name= table_name,
        auto_create_table=False  
    )

