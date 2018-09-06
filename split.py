import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import Parser
args = Parser('settings')
root = args.data_dir

def write(data, fname, root=root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))


hgg = os.listdir(os.path.join(root, 'HGG'))
hgg = [os.path.join('HGG', f) for f in hgg]

lgg = os.listdir(os.path.join(root, 'LGG'))
lgg = [os.path.join('LGG', f) for f in lgg]

X = hgg + lgg
Y = [1]*len(hgg) + [0]*len(lgg)

write(X, 'all.txt')

X, Y = np.array(X), np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
    train_list = list(X[train_index])
    valid_list = list(X[valid_index])

    write(train_list, 'train_{}.txt'.format(k))
    write(valid_list, 'valid_{}.txt'.format(k))


test = os.listdir(os.path.join(args.test_data_dir))
test = [f for f in test if not (f.endswith('.csv') or f.endswith('.txt'))]
write(test, 'test.txt', root=args.test_data_dir)
