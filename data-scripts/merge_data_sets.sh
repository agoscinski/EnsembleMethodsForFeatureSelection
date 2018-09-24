#!/bin/bash

for DATA_SET in "arcene" "dexter" "dorothea"
do
    cd "${DATA_SET^^}/${DATA_SET^^}"

    rm ${DATA_SET}.data
    rm ${DATA_SET}.labels

    cat ${DATA_SET}_train.data >> ${DATA_SET}.data
    cat ${DATA_SET}_valid.data >> ${DATA_SET}.data

    cat ${DATA_SET}_train.labels >> ${DATA_SET}.labels
    cat ../${DATA_SET}_valid.labels >> ${DATA_SET}.labels

    cd ../..
done


