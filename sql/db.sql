--\c project;

-- Optional
START TRANSACTION;

create table artists(
	artist_id text not null primary key,
	followers float,
	genres text,
    artist_name text,
    popularity integer
);

create table tracks(
	track_id text not null primary key,
	track_name text,
	popularity integer,
	duration_ms integer,
	explicit integer,
	artists text,
	id_artists text,
	release_date date,
	danceability float,
	energy float,
	track_key integer,
    loudness float,
    mode integer,
    speechiness float,
    acousticness float,
    instrumentalness float,
    liveness float,
    valence float,
    tempo float,
    time_signature integer
);

SET datestyle TO iso, ymd;

\COPY artists from 'data/artists.csv' delimiter ',' CSV header null as 'null';
\COPY tracks from 'data/tracks.csv' delimiter ',' CSV header null as 'null';

commit;

select * from tracks limit 5;
select * from artists limit 5;
