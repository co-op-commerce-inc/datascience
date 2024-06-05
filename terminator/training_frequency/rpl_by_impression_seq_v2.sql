--- Find the conversions and ad spend for each widget load by order 
with order_sessions as (
select 
    remote_order_id,
    session_id,
    min(dvce_created_tstamp) as timestamp
from event.event
where 
    to_date(dvce_created_tstamp) between '2024-01-18' and current_date - 14
group by all),
nurture_sessions as (
select distinct session_id
from event.event 
where widget_type = 'LEAD_GEN'
)
, email_sessions as (
select distinct session_id
from event.event 
where event_name = 'server_email_product_display'
)
,order_session_seq as (
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
where 
    session_id not in (select * from nurture_sessions)
    and session_id not in (select * from email_sessions)
    -- and to_date(FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC)) >= '2024-04-18'
),
session_brands as (
select 
    session_id, 
    count(distinct brand_name) as advertiser_count
from event.event e
where 
    brand_name is not null
    and brand_name != publisher_name --- do not count upsell brand impr
    and to_date(dvce_created_tstamp) >= '2024-01-18'
group by all
),
session_conv as (
--- Find events associated with a specific conversion
select 
    -- *
    e.session_id,
    -- count(distinct e.brand_name) as advertiser_count,
    count(distinct c.order_id) as conv_count,
    sum(a.billable_amount) as ad_spend
from event.event e 
left join dbt_tkaraffa.derived__conversion c using (event_id)
left join dbt_tkaraffa.curated__ad_spend_revenue a 
    on c.order_id = a.order_id
where 
    e.session_id in (
        select distinct session_id
        from event.event 
        where event_id in (
            select distinct event_id
            from dbt_tkaraffa.derived__conversion
            where 
                to_date(event_created_at) between '2024-01-18' and current_date - 14
                and conversion_type = 'cross-sell'
            )
    )
group by all
),
granular as (
select 
    remote_order_id,
    first_session_timestamp,
    session_id,
    timestamp,
    session_sequence_number,
    time_diff_minutes,
    time_diff_hours,
    time_diff_days,
    case 
      when (time_diff_hours <= 24) then 'first_day'
      when (time_diff_hours > 24 and time_diff_hours <= 84) then 'subsequent_under_half_week'
      when (time_diff_hours > 84) then 'subsequent_over_half_week'
      else 'other' end as impression_sequence,
    case when advertiser_count is null then 0 else advertiser_count end as advertiser_count,
    case when conv_count is null then 0 else conv_count end as conv_count,
    case when ad_spend is null then 0 else ad_spend end as ad_spend,
from order_session_seq 
left join session_brands sb using (session_id)
left join session_conv using (session_id)
)
-- ,date_agg as (
-- select 
--     to_date(timestamp) as date,
--     case 
--       when (time_diff_hours <= 24) then 'first_day'
--       when session_sequence_number > 1 and time_diff_hours > 24 and time_diff_hours <= 84 then 'subsequent_under_half_week'
--       when session_sequence_number > 1 and time_diff_hours > 84 then 'subsequent_over_half_week'
--       else 'other' end as impression_sequence,
--     count(distinct session_id) as loads,
--     sum(advertiser_count) as advertiser_count,
--     sum(conv_count) as conv,
--     sum(ad_spend) as ad_spend
-- from granular
-- group by all
-- order by date asc)
-- select 
--     *,
--     -- Find 7 day avg of loads partitioned by impression_sequence
--     avg(loads) over (partition by impression_sequence order by date asc rows between 6 preceding and current row) as loads_7_day_avg,
--     -- Find 7 day avg of advertiser_count partitioned by impression_sequence
--     avg(advertiser_count) over (partition by impression_sequence order by date asc rows between 6 preceding and current row) as advertiser_count_7_day_avg,
--     -- Find 7 day avg of conv partitioned by impression_sequence
--     avg(conv) over (partition by impression_sequence order by date asc rows between 6 preceding and current row) as conv_7_day_avg,
--     -- Find 7 day avg of ad_spend partitioned by impression_sequence
--     avg(ad_spend) over (partition by impression_sequence order by date asc rows between 6 preceding and current row) as ad_spend_7_day_avg 
-- from date_agg
-- ;
select 
    *
from granular
-- where 
--     remote_order_id in (
--         select distinct remote_order_id
--         from granular
--         where 
--             session_sequence_number > 1
--             and ad_spend > 0
--     )
order by remote_order_id, session_sequence_number asc
limit 500;

---QA this query: worked out the order of events for each order, and the time between each event
with relevant_orders as (
select 
    remote_order_id,
    -- session_id,
    min(dvce_created_tstamp) as order_timestamp
from event.event
group by all
having min(to_date(dvce_created_tstamp)) between '2024-01-18' and current_date - 14),
relevant_sessions as (
select 
    remote_order_id,
    session_id,
    min(dvce_created_tstamp) as session_timestamp
from event.event
where 
    remote_order_id in (select distinct remote_order_id from relevant_orders)
group by all),
order_sessions as (
select 
    remote_order_id,
    order_timestamp,
    session_id,
    session_timestamp
from relevant_orders 
inner join relevant_sessions using (remote_order_id)
),
nurture_sessions as (
select distinct session_id
from event.event 
where widget_type = 'LEAD_GEN'
)
, email_sessions as (
select distinct session_id
from event.event 
where event_name = 'server_email_product_display'
)
,order_session_seq as (
select 
    remote_order_id,
    order_timestamp,
    session_id,
    session_timestamp,
    row_number() over (partition by remote_order_id order by session_timestamp asc) as session_sequence_number,
    TIMESTAMPDIFF(MINUTE, order_timestamp, session_timestamp) AS time_diff_minutes,
    TIMESTAMPDIFF(HOUR, order_timestamp, session_timestamp) AS time_diff_hours,
    TIMESTAMPDIFF(DAY, order_timestamp, session_timestamp) AS time_diff_days
from order_sessions 
where 
    -- remote_order_id in (select distinct remote_order_id from relevant_orders)
    session_id not in (select * from nurture_sessions)
    and session_id not in (select * from email_sessions)
    -- and to_date(FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC)) >= '2024-04-18'
)
-- ,
-- session_brands as (
-- select 
--     session_id, 
--     count(distinct brand_name) as advertiser_count
-- from event.event e
-- where 
--     brand_name is not null
--     and brand_name != publisher_name --- do not count upsell brand impr
--     and to_date(dvce_created_tstamp) >= '2024-01-18'
-- group by all
-- )
-- ,
-- session_conv as (
-- --- Find events associated with a specific conversion
-- select 
--     -- *
--     e.session_id,
--     -- count(distinct e.brand_name) as advertiser_count,
--     count(distinct c.order_id) as conv_count,
--     sum(a.billable_amount) as ad_spend
-- from event.event e 
-- left join dbt_tkaraffa.derived__conversion c using (event_id)
-- left join dbt_tkaraffa.curated__ad_spend_revenue a 
--     on c.order_id = a.order_id
-- where 
--     e.session_id in (
--         select distinct session_id
--         from event.event 
--         where event_id in (
--             select distinct event_id
--             from dbt_tkaraffa.derived__conversion
--             where 
--                 to_date(event_created_at) between '2024-01-18' and current_date - 14
--                 and conversion_type = 'cross-sell'
--             )
--     )
-- group by all
-- )
select 
    to_date(order_timestamp) as order_date,
    count(distinct remote_order_id) as order_count,
    count(distinct session_id) as session_count,
from order_session_seq
group by 1
order by 1 asc;

---QA Pt 2: figure out why the ad spend and conv count are not being calculated correctly
with relevant_orders as (
select 
    remote_order_id,
    -- session_id,
    min(dvce_created_tstamp) as order_timestamp
from event.event
group by all
having min(to_date(dvce_created_tstamp)) between '2024-02-01' and current_date - 14),
relevant_sessions as (
select 
    remote_order_id,
    session_id,
    min(dvce_created_tstamp) as session_timestamp
from event.event
where 
    remote_order_id in (select distinct remote_order_id from relevant_orders)
group by all),
order_sessions as (
select 
    remote_order_id,
    order_timestamp,
    session_id,
    session_timestamp
from relevant_orders 
inner join relevant_sessions using (remote_order_id)
),
nurture_sessions as (
select distinct session_id
from event.event 
where widget_type = 'LEAD_GEN'
)
, email_sessions as (
select distinct session_id
from event.event 
where event_name = 'server_email_product_display'
)
,order_session_seq as (
select 
    remote_order_id,
    order_timestamp,
    session_id,
    session_timestamp,
    row_number() over (partition by remote_order_id order by session_timestamp asc) as session_sequence_number,
    TIMESTAMPDIFF(MINUTE, order_timestamp, session_timestamp) AS time_diff_minutes,
    TIMESTAMPDIFF(HOUR, order_timestamp, session_timestamp) AS time_diff_hours,
    TIMESTAMPDIFF(DAY, order_timestamp, session_timestamp) AS time_diff_days
from order_sessions 
where 
    -- remote_order_id in (select distinct remote_order_id from relevant_orders)
    session_id not in (select * from nurture_sessions)
    and session_id not in (select * from email_sessions)
    -- and to_date(FIRST_VALUE(timestamp) OVER (PARTITION BY remote_order_id ORDER BY timestamp ASC)) >= '2024-04-18'
),
session_brands as (
select 
    session_id, 
    count(distinct brand_name) as advertiser_count
from event.event e
where 
    brand_name is not null
    and brand_name != publisher_name --- do not count upsell brand impr
    and to_date(dvce_created_tstamp) >= '2024-02-01'
group by all
),
relevant_event as (
select 
    session_id,
    event_id
from event.event
where 
    session_id in (select distinct session_id from order_session_seq)
),
-- exclude_orders as ( -- Filter out the new test conversion Tom is testing
-- select 
--     distinct c.order_id,
-- from dbt_tkaraffa.derived__conversion c 
-- left join dbt_tkaraffa.curated__ad_spend_revenue a using (order_id)
-- where 
--     c.attribution_method = 'device_identifier_hash' 
--     and
--     timediff('hour', c.event_created_at, c.order_created_at) > 6
-- ),
session_conv as (
select 
    -- *
    e.session_id,
    count(distinct c.order_id) as conv_count,
    sum(a.billable_amount) as ad_spend
from playground_analytics.derived__conversion_ordermap_fix c
inner join relevant_event e using (event_id)
inner join playground_analytics.curated__ad_spend_revenue_ordermap_fix a
    on c.order_id = a.order_id
-- where 
--     c.order_id not in (select * from exclude_orders)
group by all
),
granular as (
select 
    os.*,
    case 
      when (time_diff_hours <= 24) then 'first_day'
      when (time_diff_hours > 24 and time_diff_hours <= 84) then 'subsequent_under_half_week'
      when (time_diff_hours > 84) then 'subsequent_over_half_week'
      else 'other' end as impression_sequence,
    case when advertiser_count is null then 0 else advertiser_count end as advertiser_count,
    case when conv_count is null then 0 else conv_count end as conv_count,
    case when ad_spend is null then 0 else ad_spend end as ad_spend
from order_session_seq os
left join session_conv sc using (session_id)
left join session_brands sb using (session_id)
),
agg as (
select 
    to_date(order_timestamp) as order_date,
    impression_sequence,
    count(distinct remote_order_id) as order_count,
    count(distinct session_id) as load_count,
    sum(advertiser_count) as advertiser_count,
    sum(conv_count) as conv_count,
    sum(ad_spend) as ad_spend
from granular
group by all
order by 1, 2 asc)
select 
    *,
    -- Add 7 day avg of load_count partitioned by impression_sequence
    avg(load_count) over (partition by impression_sequence order by order_date asc rows between 6 preceding and current row) as load_count_7_day_avg,
    -- Add 7 day avg of conv_count partitioned by impression_sequence
    avg(conv_count) over (partition by impression_sequence order by order_date asc rows between 6 preceding and current row) as conv_count_7_day_avg,
    -- Add 7 day avg of ad_spend partitioned by impression_sequence
    avg(ad_spend) over (partition by impression_sequence order by order_date asc rows between 6 preceding and current row) as ad_spend_7_day_avg
from agg;


--- 70843	1074100.47