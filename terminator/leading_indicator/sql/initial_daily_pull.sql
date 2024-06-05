--- Track 24-hour conversions against final RPL
with loads as (
select 
    to_date(dvce_created_tstamp) as date,
    -- widget_type,
    -- ml_model,
    count(distinct event_id) as loads,
from event.event e
where 
    event_name = 'widget_display'
    and to_date(dvce_created_tstamp) between '2024-01-01' and current_date - 14
group by all 
order by date asc
),
zeroday_conv as (
--- Find 24-hour conversions
select 
    to_date(event_created_at) as date,
    count(distinct c.order_id) as zeroday_conv,
    sum(billable_amount) as zeroday_adspend,
from dbt_tkaraffa.derived__conversion c 
left join dbt_tkaraffa.curated__ad_spend_revenue using (order_id)
where
    conversion_type = 'cross-sell'
    and days_to_attribution < 1
    and to_date(event_created_at) between '2024-01-01' and current_date - 14
group by all
),
zeroday_new_cust as (
--- Find 24-hour conversions
select 
    to_date(event_created_at) as date,
    count(distinct c.order_id) as zeroday_new_conv,
    sum(billable_amount) as zeroday_new_adspend,
from dbt_tkaraffa.derived__conversion c 
left join dbt_tkaraffa.curated__ad_spend_revenue using (order_id)
where
    conversion_type = 'cross-sell'
    and days_to_attribution < 1
    and to_date(event_created_at) between '2024-01-01' and current_date - 14
    and customer_type = 'new'
group by all
),
all_conv as (
--- Find All conversions
select 
    to_date(event_created_at) as date,
    count(distinct c.order_id) as all_conv,
    sum(billable_amount) as all_ad_spend,
from dbt_tkaraffa.derived__conversion c
left join dbt_tkaraffa.curated__ad_spend_revenue using (order_id)
where
    conversion_type = 'cross-sell'
    and to_date(event_created_at) between '2024-01-01' and current_date - 14
group by all)
select 
    *,
    round(zeroday_new_conv / zeroday_conv,2) as zeroday_new_conv_percent,
    round(zeroday_adspend / loads,3) as zeroday_rpl,
    round(zeroday_new_adspend / loads,3) as zeroday_new_rpl,
    round(all_ad_spend / loads,3) as all_rpl
from loads l
left join zeroday_conv z using (date)
left join zeroday_new_cust znc using (date)
left join all_conv a using (date)
order by date asc;

-- Pull 24 clicks: identify for each load / session_id, if there was a click within 24 hours
with loads as (
select 
    session_id,
    min(to_date(dvce_created_tstamp)) as date,
from event.event e
where 
    event_name = 'widget_display'
    and to_date(dvce_created_tstamp) between '2024-01-01' and current_date - 14
group by all
),
clicks as (
select 
    session_id,
    min(to_date(dvce_created_tstamp)) as date,
    count(distinct event_id) as clicks,
from event.event e
where       
    session_id in (select session_id from loads)
    and event_name ilike '%click%'
group by all)
select 
    date,
    count(distinct session_id) as loads,
    sum(case when clicks > 0 then 1 else 0 end) as clicked,
    round(sum(case when clicks > 0 then 1 else 0 end) / count(distinct session_id)::numeric,3) as ctr
from loads l
left join clicks c using (session_id, date)
group by all
order by date asc
limit 50;

