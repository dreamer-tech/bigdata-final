SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

SET hive.enforce.bucketing=true;

use projectdb;

drop table if exists artists_part;
drop table if exists tracks_part;

create external table artists_part(
        artist_id varchar(1024),
        followers float,
        genres varchar(1024),
        artist_name varchar(1024),
        popularity int
) 	clustered by (artist_id) into 5 buckets
	stored as avro location '/project/projectdata/artists_part'
	tblproperties ('AVRO.COMPRESS'='SNAPPY');

create external table tracks_part(
        track_id varchar(1024),
        track_name varchar(1024),
        popularity int,
        duration_ms int,
        explicit int,
        artists varchar(1024),
        id_artists varchar(1024),
        release_date date,
        danceability float,
        energy float,
        track_key int,
        loudness float,
        mode int,
        speechiness float,
        acousticness float,
        instrumentalness float,
        liveness float,
        valence float,
        tempo float,
        time_signature int
) 	clustered by (track_id) into 5 buckets
	stored as avro location '/project/projectdata/tracks_part'
        tblproperties ('AVRO.COMPRESS'='SNAPPY');

insert into artists_part SELECT * FROM artists;
insert into tracks_part SELECT track_id, track_name, popularity, duration_ms, explicit, artists, id_artists, cast(to_date(from_utc_timestamp(release_date, "+00")) as date), danceability, energy, track_key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature FROM tracks;