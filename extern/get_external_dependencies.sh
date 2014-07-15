#!/bin/bash
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/resources/MCG-PreTrained.tgz
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip -O liblinear.zip
mkdir liblinear
unzip liblinear.zip -d liblinear
cd liblinear/liblinear-1.94
make
cd ../..
tar -zxvf MCG-PreTrained.tgz
echo "You need to change the path in MCG-PreTrained/root_dir.m!" 
