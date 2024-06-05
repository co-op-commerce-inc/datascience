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
# Filter out all warnings
warnings.filterwarnings('ignore', category=Warning)


### Version 1 Queries
raw_daily_display_query = """
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
    and to_date(event_created_at) between '2024-02-29' and current_date -1
    -- and ml_model is null
    and brand_name != publisher_name
group by all 
order by event_date asc
"""

raw_daily_display_click_query = """
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
    and to_date(event_created_at) between '2024-02-29' and current_date -1
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
    and to_date(event_created_at) between '2024-02-29' and current_date -1
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

raw_zero_day_adspend_query = """
with conv_table as (
select *
from playground_analytics.derived__conversion_ordermap_fix 
where to_date(order_created_at) between '2024-02-01' and '2024-03-31'
union all 
select *
from derived.conversion 
where to_date(order_created_at) between '2024-04-01' and current_date -1
)
select 
    to_date(c.event_created_at) as event_date,
    case 
        when e.widget_type = 'DISCOFEED' and e.ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
        when e.ml_model ilike '%waterfall%' then 'waterfall'
        when e.widget_type = 'LEAD_GEN' then 'nurture'
        when e.ml_model = 'contextual' and e.widget_type != 'LEAD_GEN' then 'contextual_classic'
        else 'other' 
    end as load_type,
    sum(case when billable_amount is null then 0 else billable_amount end) as zeroday_ad_spend
from conv_table c
left join curated.ad_spend_revenue asr using (order_id)
left join event.event e on e.event_id = c.event_id
where 
    c.conversion_type = 'cross-sell'
    and to_date(c.event_created_at) between '2024-02-29' and current_date -1
    and c.days_to_attribution = 0
group by all 
order by event_date asc
;
"""


final_conv_query = """
with conv_table as (
select *
from playground_analytics.derived__conversion_ordermap_fix 
where to_date(order_created_at) between '2024-02-01' and '2024-03-31'
union all 
select *
from derived.conversion 
where to_date(order_created_at) between '2024-04-01' and current_date -1
)
select 
    to_date(c.event_created_at) as event_date,
        -- case 
    --     when e.widget_type = 'DISCOFEED' and e.ml_model not ilike '%waterfall%' and e.ml_model != 'contextual' then 'bert_boost_classic'
    --     when e.ml_model ilike '%waterfall%' then 'waterfall'
    --     when e.widget_type = 'LEAD_GEN' then 'nurture'
    --     when e.ml_model = 'contextual' and e.widget_type != 'LEAD_GEN' then 'contextual_classic'
    --     else 'other' 
    -- end as load_type,
    count(distinct c.order_id) as final_conv_count,
    sum(case when billable_amount is null then 0 else billable_amount end) as final_ad_spend,
    -- count(case when c.customer_type = 'new' then 1 else null end) / count(distinct c.order_id) as new_cust_rate,
    -- count(case when c.event_classification = 'click' then 1 else null end) / count(distinct c.order_id) as click_rate,
from event.event e 
left join conv_table c using (event_id)
left join curated.ad_spend_revenue asr on c.order_id = asr.order_id
where 
    c.conversion_type = 'cross-sell'
    and to_date(c.event_created_at) between '2024-02-29' and current_date -1
    -- and c.days_to_attribution = 0
group by all 
order by event_date asc
;
"""

daily_order_query = """
select 
    to_date(created_at_gmt) as event_date,
    count(distinct order_id) as order_count
from orders.order_data_flattened
where 
    to_date(created_at_gmt) between '2024-02-29' and current_date-1
group by all
order by event_date asc
;
"""

### Version 2 Queries

#Create a brand-level display and zeroday conversion query
