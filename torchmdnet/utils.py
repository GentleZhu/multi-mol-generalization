import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from pytorch_lightning.utilities import rank_zero_warn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

path = "/shared/data3/shivama2/pre-training-via-denoising/data/qm9/raw/gdb9.sdf"
iid_split_proto = False
weighted_proto = False


def train_val_test_split(dataset, train_size, val_size, test_size, seed, order=None):
    dset_len = len(dataset)
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")



    idxs = np.arange(dset_len, dtype=np.int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def get_scaffold(smile):

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False) #=Chem.MolFromSmiles(smile)
    return scaffold

def get_domain_sorted_list(data_list,dslen,train_size,val_size,names_list,seed,order):
    scaffolds = defaultdict(list)
    weights = defaultdict()
    ind = -1
    for data in data_list:
            # print(data)
            ind = ind + 1
            # print(names_list[ind], ind)
            # exit(0)
            try: 
                smile = Chem.MolToSmiles(data)
                scaff = get_scaffold(smile)
                scaffolds[scaff].append(ind)
            except ValueError as e:
                scaffolds['unkown'].append(ind)
    for sc in scaffolds.keys():
        if weighted_proto:
            w = 1-len(sc)/dslen
        else:
            w = 1
        for i in scaffolds[sc]:
            # print(names_list[i])
            # print('here-n+2')
            weights[names_list[i]] = w
    

    test_size = dslen - train_size - val_size
    if iid_split_proto:
        total = train_size + val_size + test_size
        idxs = np.arange(dslen, dtype=np.int)
        if order is None:
            idxs = np.random.default_rng(seed).permutation(idxs)

        idx_train = idxs[:train_size]
        idx_val = idxs[train_size : train_size + val_size]
        idx_test = idxs[train_size + val_size : total]

        if order is not None:
            idx_train = [order[i] for i in idx_train]
            idx_val = [order[i] for i in idx_val]
            idx_test = [order[i] for i in idx_test]

        return np.array(idx_train), np.array(idx_val), np.array(idx_test),weights
    
    shuffler = list(scaffolds.keys())
    
    np.random.shuffle(shuffler)
    # print(shuffler)
    # train_size = 110000 
    # val_size = 10000
      
    idx_train = []
    idx_val = []
    idx_test = []
    visited = set()
    for index, i in enumerate(shuffler): 
        if (len(scaffolds[i]) + len(idx_test) <= test_size) and (i not in visited):
            idx_test.extend(scaffolds[i])
            # idx_test = [item for sublist in idx_test for item in sublist]
            visited.add(i)
        #    shuffler.remove(scaffolds[])

    for index, i in enumerate(shuffler):
        if (len(scaffolds[i]) + len(idx_val) <= val_size) and (i not in visited):
            idx_val.extend(scaffolds[i])
            # idx_val = [item for sublist in idx_val for item in sublist]
            visited.add(i)
    
    for index, i in enumerate(shuffler):
        #  if len(idx_test) < test_size:
        if i not in visited:
            idx_train.extend(scaffolds[i])
            # idx_train = [item for sublist in idx_train for item in sublist]
            visited.add(i)

    flat_list_test = idx_test #[item for sublist in idx_test for item in sublist]
    flat_list_val = idx_val #[item for sublist in idx_val for item in sublist]
    flat_list_train = idx_train #[item for sublist in idx_train for item in sublist]

    # print(flat_list_test)
    print(len(flat_list_val))
    print(len(flat_list_train))
    print(len(flat_list_test))
    print(dslen)
    # print(max(flat_list_train))
    # print(max(flat_list_test))
    # print(max(flat_list_val))
    print(len(flat_list_test)+len(flat_list_train)+len(flat_list_val))
    # print(min(flat_list_train))
    # print(min(flat_list_test))
    # print(min(flat_list_val))
    idx_train=np.array(flat_list_train)
    idx_val=np.array(flat_list_val)
    idx_test=np.array(flat_list_test)
    # print(idx_test.shape)
    return idx_train, idx_val, idx_test, weights

def train_val_test_split2(dataset, train_size, val_size, test_size, mols,names_list,seed,order):
    dset_len = len(dataset)
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idx_train, idx_val, idx_test,weights = get_domain_sorted_list(mols,dset_len,train_size,val_size,names_list,seed,order)
    # exit()
    # idxs = np.arange(dset_len, dtype=np.int)
    # if order is None:
    #     idxs = np.random.default_rng(seed).permutation(idxs)

    # idx_train = idxs[:train_size]
    # idx_val = idxs[train_size : train_size + val_size]
    # idx_test = idxs[train_size + val_size : total]

    # if order is not None:
    #     idx_train = [order[i] for i in idx_train]
    #     idx_val = [order[i] for i in idx_val]
    #     idx_test = [order[i] for i in idx_test]

    return idx_train, idx_val, idx_test, weights



def make_splits(
    dataset,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,
    splits=None,
    order=None,
):
    dataset_len = len(dataset)

    if splits is not None:
        print('its working fine w split param')
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:

        data_list = []
        names_list = []
        # print('here-new')
        for i, data in enumerate(dataset):
            d = int(data.name.split('_')[1])
            # exit()
            data_list.append(d)
            names_list.append(data.name)
        print(len(data_list))
    
        suppl = Chem.SDMolSupplier(path, removeHs=False,
                                   sanitize=False)

        data_mols = []
        for i, data in enumerate(suppl):
            if i+1 in data_list:
                data_mols.append(data)
    
        

        # idx_train, idx_val, idx_test = train_val_test_split(
        #     dataset, train_size, val_size, test_size, seed, order
        # )
        idx_train, idx_val, idx_test,weights = train_val_test_split2(
            dataset, train_size, val_size, test_size, data_mols,names_list,seed,order
        )
     

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
        weights
    )


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass
