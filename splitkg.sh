#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
sparsity_nodes=$3
exp2=$4
sparsity_edges=$5

if [ -d "$data_dir-rich" ]; then
    rm -r "$data_dir-rich"
fi
if [ -d "$data_dir-sparse" ]; then
    rm -r "$data_dir-sparse"
fi

cp -r "$data_dir" "$data_dir-rich"
cp -r "$data_dir" "$data_dir-sparse"
cmd="python3 -m src.splitkg \
    --data_dir $data_dir \
    --sparsity_nodes $sparsity_nodes
    --sparsity_edges $sparsity_edges"

echo "Executing $cmd"

$cmd