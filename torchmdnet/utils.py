import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
from pytorch_lightning.utilities import rank_zero_warn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from IPython import embed
from rdkit.Chem import BRICS, Recap
from collections import defaultdict

path = "data/qm9/raw/gdb9.sdf"



def train_val_test_split_iid(dataset, train_size, val_size, test_size, seed, order=None):
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

def get_scaffold(mol):

    #scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False) #=Chem.MolFromSmiles(smile)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) #=Chem.MolFromSmiles(smile)
    return scaffold

def get_motif_vocab(mol_motif_dict, vocab_size=1000):
    motif_freq = defaultdict(int)
    for _, mol_motif in mol_motif_dict.items():
        for motif in mol_motif:
            motif_freq[motif] += 1
    motif_vocab = set([pair[0]  for pair in sorted(motif_freq.items(), key=lambda x:x[1], reverse=True)[:vocab_size]])
    #embed()
    mol_motif_list = defaultdict(list)
    for idx in mol_motif_dict:
        for mol_motif in mol_motif_dict[idx]:
            if mol_motif in motif_vocab:
                mol_motif_list[idx].append(mol_motif)
    return mol_motif_list, motif_vocab

def get_domain_sorted_list(data_list,dslen,train_size,val_size):
    scaffolds = defaultdict(list)
    ind = -1
    
    substructures = dict()
    for ind, mol in enumerate(data_list):
            if mol is not None:
                # get the motifs
                substructures[ind] = BRICS.BRICSDecompose(mol)
                scaff = get_scaffold(mol)
                scaffolds[scaff].append(ind)
            #except ValueError as e:
            else:
                scaffolds['unknown'].append(ind)
    # TODO: attach mol_motif to dataset
    mol_motif_list, motif_vocab = get_motif_vocab(substructures)
    #breakpoint()
    shuffler = list(scaffolds.keys())
    np.random.shuffle(shuffler)

    test_size = dslen - train_size - val_size  
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

    # check motif coverage
    '''
    known_motifs = set()
    for _idx in idx_train:
        for motif in mol_motif_list[_idx]:
            known_motifs.add(motif)
    
    number_unseen_motifs, number_test_motifs = 0, 0
    for _idx in idx_test:
        if len(mol_motif_list[_idx]) > 0:
            number_test_motifs += 1
            for motif in mol_motif_list[_idx]:
                if motif not in known_motifs:
                    number_unseen_motifs += 1
                    break
            
    embed()
    '''

    idx_train=np.array(flat_list_train)
    idx_val=np.array(flat_list_val)
    idx_test=np.array(flat_list_test)
    # print(idx_test.shape)
    return idx_train, idx_val, idx_test

def train_val_test_split_scaffold(dataset, train_size, val_size, test_size, mols):
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
    print('scaffold split')
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

    idx_train, idx_val, idx_test = get_domain_sorted_list(mols,dset_len,train_size,val_size)
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

    return idx_train, idx_val, idx_test



def make_splits(
    dataset,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,
    splits=None,
    order=None,
    split_protocol='random'
):
    dataset_len = len(dataset)

    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:

        data_list = []
        
        for i, data in enumerate(dataset):
            d = int(data.name.split('_')[1])
            # exit()
            data_list.append(d)
        suppl = Chem.SDMolSupplier(path, removeHs=True,sanitize=True)
        
        #breakpoint()
        data_mols = []
        
        #638 invalid ones, None
        data_list = set(data_list)
        for i, data in enumerate(suppl):
            if i+1 in data_list: 
                data_mols.append(data)
        #embed()
        if split_protocol == 'random':
            idx_train, idx_val, idx_test = train_val_test_split_iid(
                dataset, train_size, val_size, test_size, seed, order
            )
        elif split_protocol == 'scaffold':
            idx_train, idx_val, idx_test = train_val_test_split_scaffold(
                dataset, train_size, val_size, test_size, data_mols
            )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
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
