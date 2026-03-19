#!/bin/bash

mkdir -p datasets
cd datasets

wget https://cvg-data.inf.ethz.ch/DROID-W/YouTube.zip
unzip YouTube.zip
rm YouTube.zip