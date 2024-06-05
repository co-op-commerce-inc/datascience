with order_sessions as (
select 
    remote_order_id,
    session_id,
    min(dvce_created_tstamp) as timestamp
from event.event
where 
    to_date(dvce_created_tstamp) >= '2024-01-01'
group by all),
nurture_sessions as (
select distinct session_id
from event.event 
where widget_type = 'LEAD_GEN'
),
order_session_seq as (
select 
    remote_order_id,
    session_id,
    timestamp,
    row_number() over (partition by remote_order_id order by timestamp asc) as session_sequence_number,
    FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC) AS first_session_timestamp, --- timestamp for 1st session
    TIMESTAMPDIFF(MINUTE, FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC), timestamp) AS time_diff_minutes,
    TIMESTAMPDIFF(HOUR, FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC), timestamp) AS time_diff_hours,
    TIMESTAMPDIFF(DAY, FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC), timestamp) AS time_diff_days-- time diff
from order_sessions 
where session_id not in (select * from nurture_sessions)
),
session_brands as (
select 
    session_id, 
    count(distinct brand_name) as brand_count
from event.event e
where 
    brand_name is not null
    and brand_name != publisher_name --- do not count upsell brand impr
    and to_date(dvce_created_tstamp) >= '2024-01-01'
group by all
),
session_conv as ( --- counts the conversion metrics at the session level
select 
    session_id,
    -- c.event_brand_name as brand_name,
    count(distinct c.order_id) as conv,
    sum(billable_amount) as ad_spend
from derived.conversion c
left join curated.ad_spend_revenue asr 
    on c.order_id = asr.order_id 
left join event.event e
    on e.event_id = c.event_id
where
    to_date(e.dvce_created_tstamp) >= '2024-01-01'
    and c.conversion_type = 'cross-sell'
group by all
),
granular as (
select 
    remote_order_id,
    timestamp,
    session_id,
    session_sequence_number,
    time_diff_minutes,
    time_diff_hours,
    time_diff_days,
    case when brand_count is null then 0 else brand_count end as brand_count,
    case when conv is null then 0 else conv end as conv,
    case when ad_spend is null then 0 else ad_spend end as ad_spend,
from order_session_seq 
left join session_brands sb using (session_id)
left join session_conv using (session_id)
)
select 
    to_date(timestamp) as date,
    case 
      when (session_sequence_number = 1 or time_diff_minutes <= 10) then 'first'
      when session_sequence_number > 1 and time_diff_minutes > 10 and time_diff_hours <= 84 then 'subsequent_under_half_week'
      when session_sequence_number > 1 and time_diff_hours > 84 then 'subsequent_over_half_week'
      else 'other' end as impression_sequence,
    count(distinct remote_order_id) as unique_orders,
    sum(brand_count) as brand_count,
    sum(conv) as conv,
    sum(ad_spend) as ad_spend
from granular
group by all
order by date asc;