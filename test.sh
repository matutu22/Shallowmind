#!/bin/bash

myTeam=teams/ShallowMind/myTeam
enemy=teams/ShallowMind/myTeam

mkdir test_data

folder=test_data/$(date +"%m%d_%H%M%S")/

mkdir $folder

cp test.sh $folder

for i in {11..20}
do
    echo "testing $i"
    echo "map RANDOM$i" >> "$folder""result.txt"
    python capture.py -r $myTeam -b $enemy -q -l RANDOM$i >> "$folder""result.txt"
done    
