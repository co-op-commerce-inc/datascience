--- What is our RPL and CVR by ad slot?

with conv_events as (
select 
    to_date(asr.event_created_at) as event_date,
    to_date(asr.order_created_at) as order_date,
    days_to_attribution,
    customer_type,
    session_id,
    e.widget_type,
    e.ml_model,
    e.event_id,
    publisher_name,
    brand_name,
    vertical_display_position,
    horizontal_display_position,
    event_name,
    event_classification,
    asr.order_id, 
    asr.billable_amount,
from event.event e 
left join curated.ad_spend_revenue asr 
    on e.event_id = asr.event_id
where 
    to_date(dvce_created_tstamp) >= '2024-01-01'
    and asr.order_id is not null 
    and publisher_name <> brand_name
    and conversion_type = 'cross-sell'
)
;

select *
from event.event 
where to_date(dvce_created_tstamp) = '2024-04-15'
order by session_id, dvce_created_tstamp asc
limit 50;


select 
    order_id,
    remote_order_id,
    session_id, 
    widget_type,
    ml_model,
    dvce_created_tstamp,
    publisher_name,
    event_name, 
    brand_name,
from event.event 
where 
    -- order_id = '4503698'
    -- session_id = '000023bc-2d82-41fd-a004-6ec368eed902'
    remote_order_id = '5236513439780'
-- group by all
order by dvce_created_tstamp asc
;

with order_sessions as (
select distinct session_id
from event.event 
where order_id is null and remote_order_id is not null
-- order by dvce_created_tstamp desc
limit 10)
select *
from event.event
where 
    session_id in (select * from order_sessions)
order by dvce_created_tstamp desc
limit 50
;

--- Pull brand displays
select 
    -- to_date(dvce_created_tstamp) as event_date,
    session_id,
    widget_type,
    ml_model,
    event_id,
    publisher_name,
    brand_name,
    vertical_display_position,
    horizontal_display_position,
    event_name,
    min(dvce_created_tstamp) as earliest_time_stamp
from event.event 
where 
    event_name = 'widget_brand_display'
    and to_date(dvce_created_tstamp) >= '2024-01-01'
    and publisher_name <> brand_name
group by all
order by session_id asc
limit 50;


with first_timestamp as (
select remote_order_id, min(dvce_created_tstamp) as dvce_created_tstamp
from event.event 
where to_date(dvce_created_tstamp) >= '2024-01-01'
group by all
),
first_session as (
select distinct session_id 
from event.event e 
left join first_timestamp using (remote_order_id, dvce_created_tstamp)
),
relevant_events as (
select 
    e.*, 
    asr.order_id as conv_order_id, 
    asr.billable_amount, 
    asr.conversion_type, 
    event_classification
from event.event e 
left join curated.ad_spend_revenue asr using (event_id)
where 
    session_id in (select * from first_session)
    and to_date(dvce_created_tstamp) >= '2024-01-01'
),
conv as (
select 
    publisher_name,
    brand_name, 
    ml_model,
    widget_type,
    vertical_display_position,
    -- event_classification,
    count(distinct conv_order_id) as conv,
    sum(billable_amount) as ad_spend
from relevant_events
where 
    conversion_type = 'cross-sell'
    -- and widget_type != 'LEAD_GEN'
group by all)
,
displays as (
select 
    publisher_name,
    brand_name, 
    ml_model,
    widget_type,
    vertical_display_position,
    count(distinct event_id) as brand_displays
from relevant_events
where 
    event_name = 'widget_brand_display'
    -- and widget_type != 'LEAD_GEN'
group by all),
granular as (
select 
    publisher_name, 
    brand_name,
    ml_model,
    widget_type,
    vertical_display_position,
    brand_displays,
    case when conv is null then 0 else conv end as conv,
    case when ad_spend is null then 0 else ad_spend end as ad_spend
from displays d
left join conv c 
    using (
        publisher_name, brand_name, 
        vertical_display_position,ml_model,
        widget_type
        )
where 
    vertical_display_position != '-1'
)
select 
    publisher_name, 
    brand_name,
    -- ml_model,
    widget_type,
    case 
        when vertical_display_position = 0 then '1'
        when vertical_display_position = 1 then '2'
        when vertical_display_position = 2 then '3'
        else 'below fold' 
        end as display_position,
    sum(brand_displays) as brand_displays,
    sum(conv) as conv,
    sum(ad_spend) as ad_spend,
from granular
group by all
;