#!/usr/bin/env bash
TASK=q1
DATASET=rush4-1
nsml run -g 1 -c 2 -d ${DATASET} -a "--config_file ${TASK}/config.yaml"

# windows console 용 복붙

# nsml run -g 1 -c 2 -d rush4-1 -a "--config_file q1/config.yaml"