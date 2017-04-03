SELECT cities.city_name, percentile_cont(0.9) within GROUP 
       (ORDER BY abs(trips.predicted_eta-trips.actual_eta) ASC)  as percentile_90th

        FROM    cities LEFT JOIN trips 
            ON  cities.city_id=trips.city_id

        WHERE   cities.city_name IN ('Qarth', 'Meereen')
            AND trips.requested_at > now() - INTERVAL '30 days' 
            AND trips.status='completed'

        GROUP BY cities.city_name
        ORDER BY cities.city_name
;

WITH success as (
    SELECT  cities.city_name,  events.rider_id, events._ts as ts

      FROM  cities JOIN events
         ON cities.city_id=events.city_id

     WHERE  cities.city_name IN ('Qarth', 'Meereen')
        AND events.event_name='sign_up_success' 
        AND events._ts BETWEEN to_date('2016-01-01','YYYY-MM-DD') 
                           AND to_date('2016-01-07','YYYY-MM-DD')
    ),

WITH trip_first as (
   SELECT client_id, min(request_at) 
     FROM trips
    GROUP BY client_id
    )

SELECT success.city_name, date_part('day', success.dt) AS dow, 
       100.0 * AVG(CASE WHEN trip_first.request_at - success.ts < INTERVAL '168 hours' 
                            THEN 1 
                            ELSE 0 END) AS pct_success

    FROM success LEFT JOIN trip_first
        ON trip_first.client_id=success.rider_id

    GROUP BY success.city_name, date_part('day', success.dt)
    ORDER BY success.city_name, dow
;
