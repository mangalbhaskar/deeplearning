#!/bin/bash

DATA_PATH="data/traffic-signs/BelgiumTSC"
LOG_PATH="nnmodels/traffic-signs/BelgiumTSC"
mkdir -p $DATA_PATH $LOG_PATH

if [ ! -f $DATA_PATH/BelgiumTSC_Training.zip ]; then
  wget -c https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip -P $DATA_PATH
fi

if [ ! -f $DATA_PATH/BelgiumTSC_Testing.zip ]; then
  wget -c https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip -P $DATA_PATH
fi

if [ ! -d $DATA_PATH/BelgiumTSC_Training ]; then
  unzip -q $DATA_PATH/BelgiumTSC_Training.zip -d $DATA_PATH
fi

if [ ! -d $DATA_PATH/BelgiumTSC_Testing ]; then
  unzip -q $DATA_PATH/BelgiumTSC_Testing.zip -d $DATA_PATH
fi
