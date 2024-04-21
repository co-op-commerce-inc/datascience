with conv as (
select 
    to_date(event_created_at) as date,
    widget_type,
    count(distinct order_id) as conv,
    sum(billable_amount) as ad_spend,
from curated.combined_ad_spend_revenue asr 
where 
    conversion_type = 'cross-sell'
group by all),
loads as (
select 
    to_date(dvce_created_tstamp) as date,
    widget_type,
    count(distinct session_id) as widget_loads,
from event.event
where 
    event_name = 'widget_display'
group by all
order by date desc
),
granular as (
select 
    l.*,
    case when conv is null then 0 else conv end as conv,
    case when ad_spend is null then 0 else ad_spend end as ad_spend,
from loads l
left join conv using (date, widget_type)
where 
    date >= '2024-01-01'
    -- and widget_type is not null
)
select 
    case when widget_type is not null then widget_type
    else 'DISCOFEED' end as widget_type,
    date,
    sum(widget_loads) as widget_loads,
    sum(conv) as conv,
    sum(ad_spend) as ad_spend
from granular
group by all
order by date asc
;


select *
from curated.combined_discofeed_daily
limit 10;