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

--- Find aggregate metrics by month for one top brand
with exclude_orders as ( -- Filter out the new test conversion Tom is testing
select 
    distinct c.order_id,
from dbt_tkaraffa.derived__conversion c 
left join dbt_tkaraffa.curated__ad_spend_revenue a using (order_id)
where 
    c.attribution_method = 'device_identifier_hash' 
    and
    timediff('hour', c.event_created_at, c.order_created_at) > 6
);

-- Find click-through rate
with brand_displays as (
select 
    brand_name,
    month(event_created_at) as month,
    to_date(event_created_at) as date,
    count(distinct event_id) as displays
from event.event e 
where 
    event_name = 'widget_brand_display'
    and to_date(event_created_at) between '2024-01-01' and '2024-03-31'
    and brand_name != publisher_name
    and brand_name = 'Caden Lane' -- make this a variable
group by all
),
brand_clicks as (
select 
    brand_name,
    month(event_created_at) as month,
    to_date(event_created_at) as date,
    count(distinct event_id) as clicks,
    count(distinct session_id) as clicked_sessions,
from event.event e
where 
    event_name ilike '%click%'
    and to_date(event_created_at) between '2024-01-01' and '2024-03-31'
    and brand_name != publisher_name
    and brand_name = 'Caden Lane' -- make this a variable;
group by all
),
brand_ctr_benchmark as (
select 
    b.brand_name,
    b.month,
    sum(d.displays) as displays,
    sum(b.clicks) as clicks,
    sum(b.clicked_sessions) as clicked_sessions,
    round(sum(b.clicks) / sum(d.displays),3) as monthly_ctr,
    round(sum(b.clicked_sessions) / sum(d.displays),3) as monthly_clicked_rate,
from brand_clicks b
left join brand_displays d using (brand_name, month)
group by all
)
select 
    d.*,
    c.clicks,
    c.clicked_sessions,
    round(c.clicks / d.displays,3) as ctr,
    round(c.clicked_sessions / d.displays,3) as clicked_rate,
    -- Normalize to benchmark
    round((c.clicks / d.displays) / b.monthly_ctr*100,0) as ctr_index,
    round((c.clicked_sessions / d.displays) / b.monthly_clicked_rate*100,0) as clicked_rate_index
from brand_displays d 
left join brand_clicks c using (brand_name, month, date)
left join brand_ctr_benchmark b using (brand_name, month)
order by date asc
;



select * 
from orders
limit 50;

--- Event Level Breakdowns by Date
select 
    to_date(dvce_created_tstamp) as date,
    case 
        when event_name = 'widget_product_display' then 'product_display'
        when event_name = 'widget_brand_display' then 'brand_display'
        when event_name = 'widget_product_click' then 'product_click'
        when event_name = 'widget_brand_click' then 'brand_click'

        when event_name = 'page_view' then 'page_view'
        when event_namme = 'page_load' then 'page_load'
        when event_namme ilike '%email%' then 'email_event'
        when event_namme ilike '%extension%' then 'extension_event'
        else 'other'
    end as event_type,
    count(distinct event_id) as event_count,
from event.event    
where
    publisher_name != brand_name 
    and to_date(dvce_created_tstamp) between '2023-10-01' and current_date - 14
group by all
order by date asc;

select 
    to_date(created_at_gmt) as date, 
    case 
        when event = 'WIDGET_DISPLAY' then 'widget_display'
        when event = 'WIDGET_BRAND_DISPLAY' then 'brand_display'
        when event = 'PAGE_LOAD' then 'page_load'
        when event = 'WIDGET_LOAD' then 'widget_load'
        when event ilike '%email%' then 'email_event',
        when event ilike '%extension%' then 'extension_event',
        when event = 'CLICK' then 'product_click'
        when event = 'CLICK_BRAND' then 'brand_click'
        else 'other'
    end as event_type,
    count(distinct event_id) as event_count
from events.event_data_flattened
where 
    to_date(created_at_gmt) between '2023-10-18' and current_date - 14
group by all
order by date asc;
