--- Question 1: Identify the brands that are at near max budget, and when that limit was reached
select 
    order_brand_name, 
    max(max_budget) as max_budget,
    min(order_created_at) as reach_max_timestamp
from curated.ad_spend_revenue c
where 
    is_near_monthly_max_budget = True 
    and to_date(order_created_at) >= '2024-03-01'
    and to_date(order_created_at) < '2024-04-01'
group by all 
order by 1 asc
;

--- Question 2: For these brands, what are their performances like prior to reaching budget limits
with max_budgets as (
select 
    order_brand_name, 
    max(max_budget) as max_budget,
    min(order_created_at) as reach_max_timestamp
from curated.ad_spend_revenue c
where 
    is_near_monthly_max_budget = True 
    and to_date(order_created_at) >= '2024-03-01'
    and to_date(order_created_at) < '2024-04-01'
group by all 
order by 1 asc
),
widget_loads as (
SELECT 
    brand_name, 
    count(distinct session_id) AS brand_impr
FROM event.event e 
left join max_budgets mb 
    on e.brand_name = mb.order_brand_name
WHERE 
    to_date(dvce_created_tstamp) >= '2024-03-01'
    and dvce_created_tstamp <= reach_max_timestamp
    and publisher_name <> brand_name
    and event_name = 'widget_brand_display'
GROUP BY ALL
-- ORDER BY DATE ASC
),
brand_conv as (
select 
    asr.order_brand_name as brand_name,
    count(distinct order_id) as conv_count,
    sum(billable_amount) as total_ad_spend,
from curated.ad_spend_revenue asr 
left join max_budgets mb using (order_brand_name)
where 
    mb.order_brand_name is not null 
    and order_created_at <= mb.reach_max_timestamp 
    and to_date(order_created_at) >= '2024-03-01'
    and conversion_type = 'cross-sell'
group by all 
)
select 
    *,
    round(total_ad_spend / brand_impr * 1000, 2) as eCPM,
    round(conv_count / brand_impr, 4) as CvR,
    round(total_ad_spend / conv_count, 2) as blended_CPA
from widget_loads wl 
left join brand_conv bc using (brand_name)
order by total_ad_spend desc;

--- Question 3: Find the publishers, and other brands that frequently show up with this brand 
with max_budgets as (
select 
    order_brand_name, 
    max(max_budget) as max_budget,
    min(order_created_at) as reach_max_timestamp
from curated.ad_spend_revenue c
where 
    is_near_monthly_max_budget = True 
    and to_date(order_created_at) >= '2024-03-01'
    and to_date(order_created_at) < '2024-04-01'
group by all 
order by 1 asc
),
sessions as (
select distinct session_id,
from event.event e
left join max_budgets mb 
    on e.brand_name = mb.order_brand_name
where
    to_date(dvce_created_tstamp) >= '2024-03-01'
    and dvce_created_tstamp <= reach_max_timestamp
    and publisher_name <> brand_name
order by session_id asc)
select 
    distinct session_id,
    widget_type,
    ml_model,
    publisher_name,
    dvce_created_tstamp,
    brand_name,
from event.event e 
where 
    session_id in (select * from sessions)
    and brand_name is not null 
order by session_id desc
;

--- Build a query for looking at performance before and after for a specific publisher
with loads as (
select 
    case when dvce_created_tstamp <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
    count(distinct to_date(dvce_created_tstamp)) as days,
    count(distinct event_id) as widget_loads
from event.event e 
where
    publisher_name = 'Laura Geller'
    and to_date(dvce_created_tstamp) < '2024-04-01'
    and to_date(dvce_created_tstamp) >= '2024-03-01'
    and event_name = 'widget_display'
group by all),
conv as (
select 
    case when event_created_at <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
    count(distinct order_id) as conv_count,
    sum(billable_amount) as ad_spend
from curated.ad_spend_revenue
where 
    event_brand_name = 'Laura Geller'
    and to_date(event_created_at) < '2024-04-01'
    and to_date(event_created_at) >= '2024-03-01'
group by all
)
select *
from loads
inner join conv using (period)
;

left join curated.ad_spend_revenue c using(event_id)
where
    publisher_name = 'Laura Geller'
    and dvce_created_tstamp <= '2024-03-09 18:45:53.000'
    and to_date(dvce_created_tstamp) >= '2024-03-01'
);

--- Build a query for looking at performance before and after for a co-advertiser
with loads as (
select 
    case when dvce_created_tstamp <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
    count(distinct to_date(dvce_created_tstamp)) as days,
    count(distinct event_id) as brand_impr
from event.event e 
where
    brand_name = 'Liquid I.V.'
    and to_date(dvce_created_tstamp) < '2024-04-01'
    and to_date(dvce_created_tstamp) >= '2024-03-01'
    and event_name = 'widget_brand_display'
group by all),
conv as (
select 
    case when event_created_at <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
    count(distinct order_id) as conv_count,
    sum(billable_amount) as ad_spend
from curated.ad_spend_revenue
where 
    order_brand_name = 'Liquid I.V.'
    and to_date(event_created_at) < '2024-04-01'
    and to_date(event_created_at) >= '2024-03-01'
group by all
)
select *
from loads
inner join conv using (period)
;


-- Emergent Brands Query
with loads as (
    select 
        brand_name,
        case when dvce_created_tstamp <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
        count(distinct to_date(dvce_created_tstamp)) as days,
        count(distinct event_id) as brand_impr
    from event.event e 
    where
        publisher_name = 'Laura Geller'
        and to_date(dvce_created_tstamp) < '2024-04-01'
        and to_date(dvce_created_tstamp) >= '2024-03-01'
        and event_name = 'widget_brand_display'
    group by all),
conv as (
    select 
        order_brand_name as brand_name,
        case when event_created_at <= '2024-03-09 18:45:53.000' then 'Pre-Max' else 'Post-Max' end as period,
        count(distinct order_id) as conv,
        sum(billable_amount) as ad_spend,
    from curated.ad_spend_revenue
    where
        event_brand_name = 'Laura Geller'
        and to_date(event_created_at) < '2024-04-01'
        and to_date(event_created_at) >= '2024-03-01'
        -- and event_name = 'widget_brand_display'
    group by all),
granular as (
select 
    brand_name,
    period,
    days,
    brand_impr,
    case when conv is null then 0 else conv end as conv,
    case when ad_spend is null then 0 else ad_spend end as ad_spend 
from loads l 
left join conv c using (brand_name, period)
),
pivot as (
select 
    brand_name,
    -- max(days) as days,
    max(case when period = 'Pre-Max' then round(brand_impr/days,1) else 0 end) as Pre_Daily_Impr,
    max(case when period = 'Post-Max' then round(brand_impr/days,1) else 0 end) as Post_Daily_Impr,
    max(case when period = 'Pre-Max' then round(conv/days,1) else 0 end) as Pre_Daily_Conv,
    max(case when period = 'Post-Max' then round(conv/days,1) else 0 end) as Post_Daily_Conv,
    max(case when period = 'Pre-Max' then round(ad_spend/days,1) else 0 end) as Pre_Daily_Rev,
    max(case when period = 'Post-Max' then round(ad_spend/days,1) else 0 end) as Post_Daily_Rev,
from granular
group by brand_name)
select 
    *
from pivot p
where
    pre_daily_impr > 0
    and p.post_daily_impr > 100
    and (post_daily_impr - pre_daily_impr)/nullif(pre_daily_impr,0) >= 0.5
-- order by impr_change desc
;

--- Brand Category
select 
    cb.name as brand,
    cc.name as category,
from postgres.core_brand cb 
left join postgres.core_brand_categories cbc 
    on cb.core_brand_id = cbc.brand_id
left join postgres.core_category cc
    on cc.core_category_id = cbc.category_id
where 
    active = True 
    and cb.name in ('True Botanicals')
limit 10;

select 
    -- distinct session_id
    session_id,
    publisher_name,
    -- event_id,
    event_name,
    dvce_created_tstamp,
    brand_name,
    horizontal_display_position,
    vertical_display_position
from event.event 
where 
    -- brand_name = 'Blueland' --- Variable
    to_date(dvce_created_tstamp) <= '2024-04-01' --- Variable
    and to_date(dvce_created_tstamp) >= '2024-03-01' --- Variable
    and publisher_name = 'Laura Geller' --- Variable
    and widget_type = 'DISCOFEED'
order by session_id desc, dvce_created_tstamp asc
limit 50;





--- Query to Find top 2 co-advertisers
with 
relevant_sessions as (
select 
    distinct session_id
from event.event 
where 
    brand_name = 'Blueland' --- Variable
    and to_date(dvce_created_tstamp) >= '2024-03-01' --- Variable (at some point)
    and dvce_created_tstamp <= '2024-03-29 15:43:16.566' --- Variable
    and publisher_name = 'Laura Geller' --- Variable
),
sessions as (
select 
    distinct session_id,
    -- widget_type,
    -- ml_model,
    publisher_name,
    dvce_created_tstamp as timestamp,
    brand_name,
from event.event e 
where 
    session_id in (select * from relevant_sessions)
    and brand_name is not null
    and vertical_display_position <= 2
order by session_id, timestamp asc),
distinct_brands AS (
    SELECT
        session_id,
        publisher_name,
        brand_name,
        MIN(timestamp) AS earliest_timestamp
    FROM sessions
    GROUP BY ALL
),
numbered_brands AS (
    SELECT 
        session_id,
        publisher_name,
        brand_name,
        earliest_timestamp,
        ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY earliest_timestamp) AS brand_order
    FROM distinct_brands
),
sessions_with_min_three_brands AS (
    SELECT 
        session_id
    FROM numbered_brands
    GROUP BY ALL
    HAVING COUNT(DISTINCT brand_name) >= 3
),
aggregated_sessions AS (
    SELECT 
        nb.session_id,
        nb.publisher_name,
        MIN(nb.earliest_timestamp) AS session_time,
        MAX(CASE WHEN nb.brand_order = 1 THEN nb.brand_name END) AS first_brand,
        MAX(CASE WHEN nb.brand_order = 2 THEN nb.brand_name END) AS second_brand,
        MAX(CASE WHEN nb.brand_order = 3 THEN nb.brand_name END) AS third_brand
    FROM numbered_brands nb
    JOIN sessions_with_min_three_brands ON nb.session_id = sessions_with_min_three_brands.session_id
    GROUP BY ALL
),
clean_sessions as (
SELECT 
    session_id,
    publisher_name,
    session_time,
    first_brand,
    second_brand,
    third_brand
FROM aggregated_sessions
ORDER BY session_time
limit 100)
select *
from clean_sessions
;

--- Query to Find emergent brand
select
    session_id,
    min(dvce_created_tstamp) as time_stamp,
    LISTAGG(distinct brand_name, ', ') within group (order by brand_name) as session_brands
from event.event
where 
    vertical_display_position <= 2
    and brand_name is not null 
    and dvce_created_tstamp >= '2024-03-10 14:22:44.000'
    and to_date(dvce_created_tstamp) <= '2024-04-01'
group by all 
having
    count(case when brand_name in ('Beachwaver') then 1 end) > 0
    and count(case when brand_name in ('Tula Skincare') then 1 end) > 0
limit 50;

select *
from curated.combined_discofeed_daily
limit 10;
