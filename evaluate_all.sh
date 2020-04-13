#!/bin/bash

python3 week5/technique2.py S03 c010 -frcnn -c -f > c010_filtered.txt
python3 week5/technique2.py S03 c011 -frcnn -c -f > c011_filtered.txt
python3 week5/technique2.py S03 c012 -frcnn -c -f > c012_filtered.txt
python3 week5/technique2.py S03 c013 -frcnn -c -f > c013_filtered.txt
python3 week5/technique2.py S03 c014 -frcnn -c -f > c014_filtered.txt
python3 week5/technique2.py S03 c015 -frcnn -c -f > c015_filtered.txt

python3 week5/technique2.py S03 c010 -frcnn -c > c010.txt
python3 week5/technique2.py S03 c011 -frcnn -c > c011.txt
python3 week5/technique2.py S03 c012 -frcnn -c > c012.txt
python3 week5/technique2.py S03 c013 -frcnn -c > c013.txt
python3 week5/technique2.py S03 c014 -frcnn -c > c014.txt
python3 week5/technique2.py S03 c015 -frcnn -c > c015.txt
