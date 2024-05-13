with tyler_convs as (
select distinct order_id 
from playground_analytics.derived__conversion_ordermap_fix
where 
    to_date(order_created_at) >= '2024-02-01'
)
select 
    to_date(order_created_at) as date,
    -- order_id,
    sum(case when attribution_method = 'consumer_hash' then 1 else 0 end)/count(distinct order_id) as consumer_hash,
    sum(case when attribution_method = 'cookie_hash' then 1 else 0 end) / count(distinct order_id) as cookie_hash,
    sum(case when attribution_method = 'device_identifier_hash' then 1 else 0 end) / count(distinct order_id) as device_identifier_hash,
    sum(case when attribution_method = 'direct_email_match' then 1 else 0 end) / count(distinct order_id) as direct_email_match,
    sum(case when attribution_method = 'order_mapping_hash' then 1 else 0 end) / count(distinct order_id) as order_mapping_hash,
from derived.conversion_candidate
where 
    is_within_conversion_window = True
    and conversion_type = 'cross-sell'
    and to_date(order_created_at) between '2024-02-01' and current_date
    and order_id in (select * from tyler_convs)
group by all 
order by date asc;