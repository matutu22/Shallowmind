#!/bin/bash

myTeam=teams/ShallowMind/myTeam
enemy=teams/ShallowMind/myTeamHC

mkdir test_data

folder=test_data/$(date +"%m%d_%H%M%S")/

mkdir $folder

cp test.sh $folder
cp "$myTeam"".py" $folder
cp "$enemy"".py" $folder

for i in {1..10}
do
    echo "testing $i"
    echo "map RANDOM$i" >> "$folder""result.txt"
    python capture.py -r $myTeam -b $enemy -l RANDOM$i >> "$folder""result.txt"
done    
