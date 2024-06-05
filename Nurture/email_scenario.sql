-- - Offer is Claimed
-- -> Get an email with 3 brands in it (70% open)
-- -> Every week, for two months, they will get an email with the same three brands in it (if this is 2 new brands each week, how much more impactful is this)
-- Open rate cut in half for every additional email
-- -- Lower Bound - RPM of DiscoFeed in Email Today
-- -- Upper Bound - RPM of Nurture Today

--- What is the RPM of Discofeed in Email Today? $0.0076 (RPL)
--- 1. Identify email sessions 
with 
ad_spend as (
    select *
    from playground_analytics.curated__ad_spend_revenue_ordermap_fix
    where 
        to_date(order_created_at) between '2024-02-01' and '2024-03-31'
    union all 
    select * 
    from playground_data_engineering.ad_spend_revenue_preterminator
    where 
        to_date(order_created_at) >= '2024-04-01'
),
email_sessions as (
    select 
        distinct session_id
    from event.event 
    where 
        event_created_at between '2024-03-01' and '2024-05-10'
        and event_name ilike 'server_email%'
),
-- email_brand_displays as(
--     select 
--         count(distinct event_id) as brand_displays
--     from event.event 
--     inner join email_sessions s using (session_id)
--     where 
--         event_created_at between '2024-04-01' and '2024-05-15'
--         and event_name = 'widget_brand_display'

-- ),
--- 2. Identify conversions via email
email_conversions as (
    select 
        count(distinct s.session_id) as emails,
        count(distinct order_id) as conv_count,
        sum(billable_amount) as ad_spend
    -- from playground_data_engineering.ad_spend_revenue_preterminator a 
    from ad_spend a
    left join event.event e using (event_id)
    inner join email_sessions s using (session_id)
    where  
        event_created_at between '2024-03-01' and '2024-05-10'
        and conversion_type = 'cross-sell'
)
--- 3. Calculate RPL: $0.0055 per email
select 
    -- emails,
    conv_count,
    ad_spend,
    ad_spend / (select count(distinct session_id) from email_sessions) as RPL 
from email_conversions
;

--- What is the RPM of Nurture Today? $0.012
with displays as (
    select 
        -- to_date(event_created_at) as source_date,
        count(distinct event_id) as brand_displays
    from event.event 
    where 
        event_created_at between '2024-04-01' and '2024-05-15'
        and event_name = 'widget_brand_display'
        and widget_type = 'LEAD_GEN'
),
conversions as (
    select 
        -- to_date(a.event_created_at) as source_date,
        count(a.order_id) as conv_count,
        sum(a.billable_amount) as ad_spend
    -- from playground_data_engineering.ad_spend_revenue_preterminator a 
    from curated.ad_spend_revenue a 
    left join event.event e using (event_id)
    where  
        a.event_created_at between '2024-04-01' and '2024-05-15'
        and conversion_type = 'cross-sell'
        and e.widget_type = 'LEAD_GEN'
)
select 
    conv_count,
    ad_spend,
    ad_spend / (select * from displays) as RPM
from conversions;

--- Calculate the additional revenue and RPM of offer claims and email opens 

---1. For each offer claim, assume there will be an email 