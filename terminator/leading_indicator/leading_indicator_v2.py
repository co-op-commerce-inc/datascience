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
    merged_df['day_count'] = range(1, len(merged_df)+1)

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

def scale_and_split_data(df):
    scaler = StandardScaler()
    #copy the merged_df dataframe
    scaled_df = df.copy()

    #isolate the features as all columns except 'event_date', 'rpl', 'conversion_rate', 'final_conv_count', 'final_ad_spend' 
    X = scaled_df.drop(columns=['event_date', 'rpl', 'final_conv_count', 'final_ad_spend'])
    Y = df[['rpl', 'final_ad_spend']]

    # #scale the input data
    X_scaled = scaler.fit_transform(X)
    #Append the scaled data to the scaled_df dataframe
    scaled_df[X.columns] = X_scaled
    return scaled_df, X, Y

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