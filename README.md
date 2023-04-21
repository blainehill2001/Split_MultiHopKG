# Split Multi-Hop Knowledge Graph Reasoning with Reward Shaping

This is an extension of the paper:

Xi Victoria Lin, Richard Socher and Caiming Xiong. [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568). EMNLP 2018.

<img src="http://victorialin.net/img/multihopkg.png" alt="multihopkg_architecture" width="700" class="center">

## Quick Start

### Environment variables & dependencies
#### Use Docker
Build the docker image
```
docker build -< Dockerfile -t multi_hop_kg:v1.0
```

Spin up a docker container and run experiments inside it.
```
nvidia-docker run -v `pwd`:/workspace/MultiHopKG -it multi_hop_kg:v1.0
```
*The rest of the readme assumes that one works interactively inside a container. If you prefer to run experiments outside a container, please change the commands accordingly.*

#### Manually set up 
Alternatively, you can install Pytorch (>=0.4.1) manually and use the Makefile to set up the rest of the dependencies. 
```
make setup
```

### Process data
First, unpack the data files 
```
tar xvzf data-release.tgz
```
and run the following command to preprocess the datasets.
```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`<dataset>` is the name of any dataset folder in the `./data` directory. In our experiments, the five datasets used are: `umls`, `kinship`, `fb15k-237`, `wn18rr` and `nell-995`. 
`<gpu-ID>` is a non-negative integer number representing the GPU index.

Rhe following command splits a dataset into a rich KG and sparse KG
```
./splitkg.sh configs/<dataset>.sh --sparsity_nodes <float> --sparsity_edge <float>
```
where you can insert a float between 0 and 1 to represent how sparse you want your final sparse KG to be (1 represents an indentical rich and sparse KG, 0 represents an empty sparse KG). Here, we consider the "rich" KG to be the full unmasked given dataset. Defaults to 0.5 for both.

### Train models
Then the following commands can be used to train the proposed models and baselines in the paper. By default, dev set evaluation results will be printed when training terminates.

1. Train embedding-based models on the rich KG:
```
./experiment-emb.sh configs/rich_<dataset>-<emb_model>.sh --train <gpu-ID>
```
The following embedding-based models are implemented: `distmult`, `complex`, `conve` and `plm` (prompt learning extension). 

2. Train RL models (policy gradient)
```
./experiment.sh configs/sparse_<dataset>.sh --train <gpu-ID>
```

3. Train RL models (policy gradient + reward shaping)
```
./experiment-rs.sh configs/sparse_<dataset>-rs.sh --train <gpu-ID>
```

Please ensure that under /configs you create a `<rich/sparse>_<dataset>[-<emb_model>].sh` or `<rich/sparse>_<dataset>[-rs].sh` from the original code's given KGs. 

* Note: To train the RL models using reward shaping, make sure 1) you have pre-trained the embedding-based models 2) set the file path pointers to the pre-trained embedding-based models correctly ([example configuration file](configs/umls-rs.sh)) 3) if you want to use two pre-trained embedding plm models, set model to `plms` and two file paths


### Evaluate pretrained models
To generate the evaluation results of a pre-trained model, simply change the `--train` flag in the commands above to `--inference`. These will print out the H@1, H@3, H@5, H@10, and MRR performance on both dev and test sets.

1. For Plain Policy Gradient (RL only model)
```
./experiment.sh configs/sparse_<dataset>.sh --inference <gpu-ID>
```

2. For Policy Gradient + Reward Shaping
```
./experiment-rs.sh configs/sparse_<dataset>-rs.sh --inference <gpu-ID>
```


To print the inference paths generated by beam search during inference, use the `--save_beam_search_paths` flag:
```
./experiment-rs.sh configs/<dataset>-rs.sh --inference <gpu-ID> --save_beam_search_paths
```

* Note by Original Authors for the NELL-995 dataset: 

  (In this repo, work with NELL-995 was untouched. In fact, it was entirely ignored because of the below headache it induces.)

  On this dataset we split the original training data into `train.triples` and `dev.triples`, and the final model to test has to be trained with these two files combined. 
  1. To obtain the correct test set results, you need to add the `--test` flag to all data pre-processing, training and inference commands.  
    ```
    # You may need to adjust the number of training epochs based on the dev set development.

    ./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
    ./experiment-emb.sh configs/nell-995-conve.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-rs.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-rs.sh --inference <gpu-ID> --test
    ```    
  2. Leave out the `--test` flag during development.

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).

### More on implementation details
We use mini-batch training in our experiments. To save the amount of paddings (which can cause memory issues and slow down computation for knowledge graphs that contain nodes with large fan-outs),
we group the action spaces of different nodes into buckets based on their sizes. Description of the bucket implementation can be found
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/rl/graph_search/pn.py#L193) and 
[here](https://github.com/salesforce/MultiHopKG/blob/master/src/knowledge_graph.py#L164).

## Citation
If you find the resource in this repository helpful, check out the original authors and cite them at:
```
@inproceedings{LinRX2018:MultiHopKG, 
  author = {Xi Victoria Lin and Richard Socher and Caiming Xiong}, 
  title = {Multi-Hop Knowledge Graph Reasoning with Reward Shaping}, 
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2018, Brussels, Belgium, October
               31-November 4, 2018},
  year = {2018} 
}
```
