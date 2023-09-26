SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

SET hive.enforce.bucketing=true;

use projectdb;

drop table if exists artists_part;
drop table if exists tracks_part;

create external table artists_part(
        app_id int,
        title varchar(256),
        date_release date,
        win boolean,
        positive_ratio int,
        user_reviews int,
        price_final float,
        price_original float,
        discount float,
        steam_deck boolean,
	    linux boolean,
	    mac boolean
) 	partitioned by (rating varchar(32))
	clustered by (app_id) into 5 buckets
	stored as avro location '/project/projectdata/artists_part'
	tblproperties ('AVRO.COMPRESS'='SNAPPY');

create external table tracks_part(
        user_id int,
        products int,
        reviews int
) 	clustered by (user_id) into 5 buckets
	stored as avro location '/project/projectdata/users_part'
        tblproperties ('AVRO.COMPRESS'='SNAPPY');

create external table recommendations_part(
        app_id int,
        helpful int,
        funny int,
        date_review date,
        hours float,
        user_id int,
        review_id int,
	is_recommended boolean
) 	clustered by (review_id) into 5 buckets
	stored as avro location '/project/projectdata/recommendations_part'
        tblproperties ('AVRO.COMPRESS'='SNAPPY');


insert into games_part partition (rating) SELECT app_id, title, cast(to_date(from_utc_timestamp(date_release, "+00")) as date) as date_release, win, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck, linux, mac, rating FROM games;
insert into artists_part SELECT * FROM artists;
insert into recommendations_part SELECT app_id, helpful, funny,  cast(to_date(from_utc_timestamp(`date`, "+00")) as date) as date_review, hours, user_id, review_id, is_recommended  FROM recommendations;