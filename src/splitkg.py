import random
import mmap
import os
import math

from src.parse_args import parser
from src.parse_args import args

from collections import defaultdict

def split_kg(args):
    """
    uses args.sparsity_nodes and args.sparsity_edges to determine how much of rich KG to mask.
    """
    #set seed
    random.seed(12345)
    #set directory string
    rich_data_dir, sparse_data_dir = args.data_dir + "-rich", args.data_dir + "-sparse"

    set_of_nodes = set()
    #get dictionary of all [unique edges: [their line IDs]]
    dict_of_unique_edges = defaultdict(list)
    #read train triple file from sparse_data_dir
    with open(sparse_data_dir + "/train.triples", "r+b") as f:
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        for line_num, line in enumerate(iter(map_file.readline, b"")):
            head, tail, relation = line.strip().split(b'\t')
            set_of_nodes.add(head)
            set_of_nodes.add(tail)
            dict_of_unique_edges[relation].append(line_num)
        map_file.close()
    #mask nodes
    #randomly select nodes according to sparsity
    num_samples = int((1 - args.sparsity_nodes) * len(set_of_nodes))
    set_of_nodes_to_drop = set(random.sample(list(set_of_nodes), num_samples))
    with open(sparse_data_dir + "/train.triples", "r") as f_in:
        with open(sparse_data_dir + "/train_filtered.triples", "w") as f_out:
            for line in f_in:
                head, tail, relation = line.strip().split('\t')
                # Check if the line needs to be deleted
                if head not in set_of_nodes_to_drop and tail not in set_of_nodes_to_drop:
                    f_out.write(line)

    #rename triples files
    os.remove(sparse_data_dir + "/train.triples")
    os.rename(sparse_data_dir + "/train_filtered.triples", sparse_data_dir + "/train.triples")
    
    #mask edges without losing uniqueness
    #then calculate the corresponding dict of all [unique edges: [sampled line IDs to drop]]
    set_of_line_IDs_to_drop = set()
    for edge, line_nums in dict_of_unique_edges.items():
        num_samples = len(line_nums) - math.ceil(args.sparsity_edges*len(line_nums))
        set_of_line_IDs_to_drop.update(list(random.sample(line_nums, num_samples)))
     

    with open(sparse_data_dir + "/train.triples", "r") as f_in:
        with open(sparse_data_dir + "/train_filtered.triples", "w") as f_out:
            for line_num, line in enumerate(f_in):
                if line_num not in set_of_line_IDs_to_drop:
                    f_out.write(line)
    
    #rename triples files
    os.remove(sparse_data_dir + "/train.triples")
    os.rename(sparse_data_dir + "/train_filtered.triples", sparse_data_dir + "/train.triples")




    
if __name__ == '__main__':
    split_kg(args)