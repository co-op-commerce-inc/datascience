--- Create a table that joins all the sources/versions of conversion
with prod_tables as (
select 
    to_date(c.event_created_at) as source_date,
    -- order_brand_name,
    -- c.attribution_method,
    -- c.days_to_attribution
    count(distinct order_id) as conv_count,
    sum(billable_amount) as ad_spend
from derived.conversion c 
left join curated.ad_spend_revenue asr using (order_id)
where 
    to_date(c.event_created_at) between '2024-01-01' and current_date - 14
    and conversion_type = 'cross-sell'
group by all
),
order_mapping_fixed as (
select 
    to_date(c.event_created_at) as source_date,
    count(distinct order_id) as conv_count,
    sum(billable_amount) as ad_spend
from playground_analytics.derived__conversion_ordermap_fix c 
left join playground_analytics.curated__ad_spend_revenue_ordermap_fix using (order_id)
where 
    to_date(c.event_created_at) between '2024-01-01' and current_date - 14
    and conversion_type = 'cross-sell'
group by all   
),
order_mapping_and_conversion_window as (
select 
    to_date(c.event_created_at) as source_date,
    count(distinct order_id) as conv_count,
    sum(billable_amount) as ad_spend
from dbt_twilson.derived__conversion c
left join dbt_twilson.curated__ad_spend_revenue asr using (order_id)
where 
    to_date(c.event_created_at) between '2024-01-01' and current_date - 14
    and conversion_type = 'cross-sell'
group by all   
)
-- tom_all as (
-- select 
--     to_date(c.event_created_at) as source_date,
--     count(distinct order_id) as conv_count,
--     sum(billable_amount) as ad_spend
-- from dbt_tkaraffa.derived__conversion c
-- left join dbt_tkaraffa.curated__ad_spend_revenue asr using (order_id)
-- where 
--     to_date(c.event_created_at) between '2024-01-01' and current_date - 14
--     and conversion_type = 'cross-sell'
-- group by all   
-- )
select 
    source_date,
    pd.conv_count as prod_conv_count,
    pd.ad_spend as prod_ad_spend,
    omf.conv_count as order_mapping_fixed_conv_count,
    omf.ad_spend as order_mapping_fixed_ad_spend,
    omcw.conv_count as fixed_with_conversion_window_conv_count,
    omcw.ad_spend as fixed_with_conversion_window_ad_spend
    -- tom.conv_count as tom_conv_count,
    -- tom.ad_spend as tom_ad_spend
from prod_tables pd 
left join order_mapping_fixed omf using (source_date)
left join order_mapping_and_conversion_window omcw using (source_date)
-- left join tom_all tom using (source_date)
order by source_date asc
;