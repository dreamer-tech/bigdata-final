#!/bin/bash

hdfs dfs -rm -r /project/projectdata

hdfs dfs -put -f /project/avsc/*.avsc /project/avsc

hive -f sql/db.hql > ./data/hive_results.txt

hive -f sql/partition.hql > ./data/hive_partition_results.txt
