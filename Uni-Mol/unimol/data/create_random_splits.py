import os
import sys
import shutil
import random
import argparse
from math import floor, ceil

import lmdb
import numpy as np
import pickle

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    
    dataset = []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        dataset.append(data)

    return dataset

def _write_lmdb(dataset, lmdb_path):
    """
    Write a dataset (list of Python objects) to a new LMDB file.
    :param dataset: list of data items to write
    :param lmdb_path: path to the new .lmdb directory
    """
    # Delete lmdb file if it already exists to save space
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
        map_size=int(1e9)
    )
    
    with env.begin(write=True) as txn:
        for i, data_item in enumerate(dataset):
            # Convert the Python object to bytes via pickle
            value_bytes = pickle.dumps(data_item)
            # Use an integer index (converted to bytes) as the key
            key_bytes = str(i).encode('ascii')
            txn.put(key_bytes, value_bytes)

    env.close()

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True,
                   help='[base, bbp, clintox, esol, freesolv, hiv, lipo, muv, pcba, qm7dft, qm8dft, qm9dft, sider, tox21, toxcast]',
                   choices= ["base", "bbp", "clintox", "esol", "freesolv", "hiv", "lipo", "muv", "pcba", "qm7dft", "qm8dft", "qm9dft", "sider", "tox21", "toxcast"]
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_prop', type=float, default=0.8, choices=[0.6, 0.7, 0.8])

    return p.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    task_name = args.dataset
    random.seed(args.seed)

    data_path = "/workspace/unimol/data/molecular_property_prediction"

    print(f"\nWriting to {os.path.join(data_path, f'{task_name}_random')}\n")
    
    train_lmdb = read_lmdb(os.path.join(data_path, task_name, "train.lmdb"))
    valid_lmdb = read_lmdb(os.path.join(data_path, task_name, "valid.lmdb"))
    test_lmdb = read_lmdb(os.path.join(data_path, task_name, "test.lmdb"))
    full_dataset = train_lmdb + valid_lmdb + test_lmdb    
    
    print(f"Task name: {task_name}")
    print(f"Seed: {args.seed}")
    print(f"Train proportion: {args.train_prop}\n")
    print(f"#molecues in train set: {len(train_lmdb)}")
    print(f"#molecues in valid set: {len(valid_lmdb)}")
    print(f"#molecues in test set: {len(test_lmdb)}")

    full_dataset = train_lmdb + valid_lmdb + test_lmdb
    num_molecules_in_orig_dataset = len(full_dataset)
    print(f"#molecules in total: {num_molecules_in_orig_dataset}\n")

    print("Shuffling full dataset and splitting...")
    train_cutoff = floor(args.train_prop*len(full_dataset))
    valid_cutoff = int(np.round((1-args.train_prop)*1/2*len(full_dataset)))

    random.shuffle(full_dataset)
    #random_train_lmdb = full_dataset[:len(train_lmdb)]
    #random_valid_lmdb = full_dataset[len(train_lmdb):len(train_lmdb) + len(valid_lmdb)]
    #random_test_lmdb = full_dataset[len(train_lmdb) + len(valid_lmdb):]
    random_train_lmdb = full_dataset[:train_cutoff]
    random_valid_lmdb = full_dataset[train_cutoff:train_cutoff + valid_cutoff]
    random_test_lmdb = full_dataset[train_cutoff + valid_cutoff:]
    
    print(f"#molecues in shuffled train set: {len(random_train_lmdb)}")
    print(f"#molecues in shuffled valid set: {len(random_valid_lmdb)}")
    print(f"#molecues in shuffled test set: {len(random_test_lmdb)}")
    print(f"#molecules in total (shuffled): {len(random_train_lmdb) + len(random_valid_lmdb) + len(random_test_lmdb)}\n")

    # takes too long time to compute stats since qm9 contains 133885 molecules.
    if task_name != "qm9dft": 
        indices_orig_train = [datapoint['ori_index'] for datapoint in train_lmdb]
        indices_random_train = [datapoint['ori_index'] for datapoint in random_train_lmdb]

        indices_orig_valid = [datapoint['ori_index'] for datapoint in valid_lmdb]
        indices_random_valid = [datapoint['ori_index'] for datapoint in random_valid_lmdb]

        indices_orig_test = [datapoint['ori_index'] for datapoint in test_lmdb]
        indices_random_test = [datapoint['ori_index'] for datapoint in random_test_lmdb]

        print(f"#molecules that are kept in train: {len([idx for idx in indices_orig_train if idx in indices_random_train])}")
        print(f"#molecules that are kept in valid: {len([idx for idx in indices_orig_valid if idx in indices_random_valid])}")
        print(f"#molecules that are kept in test: {len([idx for idx in indices_orig_test if idx in indices_random_test])}\n")
    
    assert len(full_dataset) == num_molecules_in_orig_dataset, "We lost some molecules along the way ..."

    random_train_path = os.path.join(f"{data_path}_random", task_name, "train.lmdb")
    random_valid_path = os.path.join(f"{data_path}_random", task_name, "valid.lmdb")
    random_test_path = os.path.join(f"{data_path}_random", task_name, "test.lmdb")

    #A = read_lmdb(random_train_path)
    #print(A[0].keys())
    #print(A[0]['atoms'] == random_train_lmdb[0]['atoms'])
    #print(all(np.array_equal(a, b) for a, b in zip(A[0]['coordinates'], random_train_lmdb[0]['coordinates'])))
    #print(A[0]['smi'] == random_train_lmdb[0]['smi'])
    #print(A[0]['scaffold'] == random_train_lmdb[0]['scaffold'])
    #print(A[0]['target'] == random_train_lmdb[0]['target'])
    #print(A[0]['ori_index'] == random_train_lmdb[0]['ori_index'])

    if not os.path.exists(os.path.join(f"{data_path}_random", task_name)):
        os.makedirs(os.path.join(f"{data_path}_random", task_name))
    
    _write_lmdb(random_train_lmdb, random_train_path)
    _write_lmdb(random_valid_lmdb, random_valid_path)
    _write_lmdb(random_test_lmdb, random_test_path)

    print("done.")













    