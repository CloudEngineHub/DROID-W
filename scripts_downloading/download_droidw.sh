#!/bin/bash

mkdir -p datasets
cd datasets

wget https://cvg-data.inf.ethz.ch/DROID-W/DROID-W.zip
unzip DROID-W.zip
rm DROID-W.zip