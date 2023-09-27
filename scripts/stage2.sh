#!/bin/bash

hdfs dfs -rm -r /project/projectdata

hdfs dfs -put -f /project/avsc/*.avsc /project/avsc

hive -f sql/db.hql > ./data/hive_results.txt

hive -f sql/partition.hql > ./data/hive_partition_results.txt

hive -f sql/eda.hql

echo "popularity_release_year_corr" > output/q1.csv
cat /root/q1/* >> output/q1.csv
cat output/q1.csv


echo "most_popular_danceability_ratio" > output/q2.csv
cat /root/q2/* >> output/q2.csv
cat output/q2.csv


echo "popularity_instrumentalness_corr" > output/q3.csv
cat /root/q3/* >> output/q3.csv
cat output/q3.csv