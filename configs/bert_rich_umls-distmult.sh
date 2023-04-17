#!/usr/bin/env bash

data_dir="data/umls-rich"
model="distmult"
add_reversed_training_edges="False"
group_examples_by_query="True"
entity_dim=400
relation_dim=400
num_rollouts=5
bucket_interval=10
num_epochs=1000
num_wait_epochs=50
batch_size=100
train_batch_size=100
dev_batch_size=32
learning_rate=0.0002
grad_norm=5
emb_dropout_rate=0.5
beam_size=128
num_negative_samples=20
margin=10
