#!/bin/bash

#
# Install script for the TWOSIDES dataset.
# This should be run in the project root directory.
#

if [ ! -f "../TWOSIDES/TWOSIDES.csv" ]
then
    echo "Downloading TWOSIDES into data dir."
    cd ../TWOSIDES
    wget http://tatonettilab.org/resources/nsides/TWOSIDES.csv.gz
    gunzip TWOSIDES.csv.gz
    rm TWOSIDES.csv.gz
else
    echo "TWOSIDES is already downloaded"
fi

