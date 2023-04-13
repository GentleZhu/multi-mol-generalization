# from munch import Munch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def get_scaffold(smile, mol):

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False) #=Chem.MolFromSmiles(smile)
    return scaffold

    
def get_domain_sorted_list(data_list,dslen):
    numnber_error=0
    scaffolds = defaultdict(list)
    ind = -1
    for data in tqdm(data_list):
            # print(data)
            ind = ind + 1
            try: 
                smile = Chem.MolToSmiles(data)
                scaff = get_scaffold(smile,data)
                # print()
                scaffolds[scaff].append(ind)
                # print(scaff)
            except ValueError as e:
                 numnber_error += 1
                 scaffolds['unkown'].append(ind)
        
    
    # print(scaffolds)
    shuffler = list(scaffolds.keys())
    # print(shuffler)
    np.random.shuffle(shuffler)
    print(shuffler)
    train_size = 110000 
    val_size = 10000
    test_size = dslen - train_size - val_size  
    idx_train = []
    idx_val = []
    idx_test = []
    visited = set()
    for index, i in enumerate(shuffler):
         if len(idx_test) < test_size:
              if len(scaffolds[i]) + len(idx_test) <= test_size and i not in visited:
                   idx_test.append(scaffolds[i])
                   visited.add(i)
                #    shuffler.remove(scaffolds[])

    for index, i in enumerate(shuffler):
         if len(idx_val) < val_size:
              if len(scaffolds[i]) + len(idx_val) <= val_size and i not in visited:
                   idx_val.append(scaffolds[i])
                   visited.add(i)
    
    for index, i in enumerate(shuffler):
        #  if len(idx_test) < test_size:
        if i not in visited:
            idx_train.append(scaffolds[i])
            visited.add(i)
    flat_list_test = [item for sublist in idx_test for item in sublist]
    flat_list_val = [item for sublist in idx_val for item in sublist]
    flat_list_train = [item for sublist in idx_train for item in sublist]

    print(flat_list_test)
    print(len(flat_list_val))
    print(len(flat_list_train))
    print(len(flat_list_test))
    print(dslen)
    print(max(flat_list_train))
    print(max(flat_list_test))
    print(max(flat_list_val))
    print(len(flat_list_test)+len(flat_list_train)+len(flat_list_val))
    print(min(flat_list_train))
    print(min(flat_list_test))
    print(min(flat_list_val))
    idx_train=np.array(flat_list_train)
    idx_val=np.array(flat_list_val)
    idx_test=np.array(flat_list_test)
    print(idx_test.shape)
    np.savez('scaffold_split', idx_train=np.array(flat_list_train), idx_val=np.array(flat_list_val), idx_test=np.array(flat_list_test))

         

    # for key, value in scaffolds.items():
        #  if 
             
    # print(numnber_error)
    # print(len(scaffolds))
    # print(len(list(set(scaffolds))))
    # print()
    # num = 0
    # for key, value in scaffolds.items():
    #      if value ==1:
    #           num+=1
    # print('number of 1',num)
         
    # print(sorted(scaffolds.values(),reverse=True))
    # plt.bar(list(scaffolds.keys()), scaffolds.values())
    # plt.savefig('plot.png')
    #         # data.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(smile))

    #         # break
    # train, test, val = None, None, None
    # return train, test, val



def splitter(path):
    suppl = Chem.SDMolSupplier(path, removeHs=False,
                                   sanitize=False)
    
    data_list = []
    for i, data in enumerate(suppl):
        data_list.append(data)
    # print(data_list)
    dslen = len(data_list) #130831#
    print(dslen)
    exit(0)
    get_domain_sorted_list(data_list,dslen)

splitter("/home/shivama2/pre-training-via-denoising/data/qm9/raw/gdb9.sdf")






