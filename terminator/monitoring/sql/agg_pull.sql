--- For each top brand, find these aggregate metrics
    --- Conv. Rate
    --- Ad Spend
    --- CPA/CPO
    --- New Cust %
    --- Click-Through Rate
--- Set these averages as index benchmarks
--- Plot each brand's daily metrics against the benchmarks

--- Find top 25 brands by total ad spend
with exclude_orders as ( -- Filter out the new test conversion Tom is testing
select 
    distinct c.order_id,
from dbt_tkaraffa.derived__conversion c 
left join dbt_tkaraffa.curated__ad_spend_revenue a using (order_id)
where 
    c.attribution_method = 'device_identifier_hash' 
    and
    timediff('hour', c.event_created_at, c.order_created_at) > 6
)
select 
    order_brand_name as brand,
    -- month(event_created_at) as month,
    count(distinct order_id) as conversions,
    sum(billable_amount) as ad_spend,
from dbt_tkaraffa.derived__conversion c 
left join dbt_tkaraffa.curated__ad_spend_revenue a using (order_id)
where 
    order_id not in (select * from exclude_orders)
    and to_date(c.event_created_at) between '2024-01-01' and '2024-03-31'
    and conversion_type = 'cross-sell'
group by all 
order by ad_spend desc
limit 25
;

--- 


