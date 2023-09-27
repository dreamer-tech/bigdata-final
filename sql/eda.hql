USE projectdb;

INSERT OVERWRITE LOCAL DIRECTORY '/root/q1'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT CORR(year(release_date), popularity) AS popularity_release_year_corr FROM tracks_part;


WITH popular_cnt AS (
  SELECT count(*) AS n FROM tracks_part WHERE popularity > 50
),
popular_and_danceable_cnt AS (
    SELECT count(*) AS n FROM tracks_part WHERE popularity > 50 AND danceability >= 0.5
)
INSERT OVERWRITE LOCAL DIRECTORY '/root/q2'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT popular_and_danceable_cnt.n / popular_cnt.n * 100 FROM popular_cnt, popular_and_danceable_cnt;


INSERT OVERWRITE LOCAL DIRECTORY '/root/q3'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT CORR(instrumentalness, popularity) AS popularity_instrumentalness_corr FROM tracks_part;