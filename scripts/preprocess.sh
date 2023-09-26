#!/bin/bash


rm -f ./data/*.zip
rm -f ./data/*.json
rm -f ./data/*.csv

fileid="1Y00kXVeymQGFPSasl03dt7ygpeiYNxDc"
filename="archive.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip -o ./${filename} -d ./data

sed -i 's/,\[/,{/' data/artists.csv
sed -i 's/\],/},/' data/artists.csv

sed -i 's/,\[/,{/' data/tracks.csv
sed -i 's/\],/},/' data/tracks.csv

rm ./cookie
rm ${filename}
rm -f ./output/*.csv
rm -f ./output/*.json
rm -f ./output/*.log