#!/bin/bash

rm -fr ./output/pda
rm -fr ./output/models

mkdir ./output/pda
mkdir ./output/models

export PYTHONIOENCODING=utf8;

spark-submit --jars /usr/hdp/current/hive-client/lib/hive-metastore-1.2.1000.2.6.5.0-292.jar,/usr/hdp/current/hive-client/lib/hive-exec-1.2.1000.2.6.5.0-292.jar --packages org.apache.spark:spark-avro_2.12:3.0.3 scripts/model.py


rm -f output/cv_gbt_config.csv
cat output/pda/cv_gbt_config/* >> output/cv_gbt_config.csv
rm -f output/best_gbt_params.csv
cat output/pda/best_gbt_params/* >> output/best_gbt_params.csv
rm -f output/best_gbt_scores.csv
cat output/pda/best_gbt_scores/* >> output/best_gbt_scores.csv
rm -f output/gbt_popularity.json
cat output/pda/gbt_popularity/* >> output/gbt_popularity.json

rm -f output/cv_rf_config.csv
cat output/pda/cv_rf_config/* >> output/cv_rf_config.csv
rm -f output/rf_features.csv
cat output/pda/rf_features/* >> output/rf_features.csv
rm -f output/best_rf_params.csv
cat output/pda/best_rf_params/* >> output/best_rf_params.csv
rm -f output/best_rf_scores.csv
cat output/pda/best_rf_scores/* >> output/best_rf_scores.csv
rm -f output/rf_popularity.json
cat output/pda/rf_popularity/* >> output/rf_popularity.json

rm -rf models/*/
mv -f output/models/* models/