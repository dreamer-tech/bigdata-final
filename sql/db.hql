DROP DATABASE IF EXISTS projectdb CASCADE;

CREATE DATABASE projectdb;
USE projectdb;

SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

CREATE EXTERNAL TABLE artists STORED AS AVRO LOCATION '/project/artists' TBLPROPERTIES ('avro.schema.url'='/project/avsc/artists.avsc');
CREATE EXTERNAL TABLE tracks STORED AS AVRO LOCATION '/project/tracks' TBLPROPERTIES ('avro.schema.url'='/project/avsc/tracks.avsc');

SELECT count(*) FROM artists;
SELECT count(*) FROM tracks;