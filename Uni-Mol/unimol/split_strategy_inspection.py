import lmdb
import numpy as np
import os
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

    scaffold_list = []
    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        scaffold_list.append(data["scaffold"])
    
    return scaffold_list

if __name__ == "__main__":
    task_name = "lipo"
    lmdb_path = f"/workspace/unimol/data/molecular_property_prediction/{task_name}/test.lmdb"

    print(read_lmdb(lmdb_path))
