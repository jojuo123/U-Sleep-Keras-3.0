import os

def count_samples(root, dataset):
    path = os.path.join(root, dataset, 'views/fixed_split')
    total = 0
    for d in ['train', 'val', 'test']:
        path_ = os.path.join(path, d, "LIST_OF_FILES.txt")
        with open(path_, 'r') as fp:
            lines = 0
            # lines = sum(1 for line in fp)
            for l in fp.readlines():
                if 'mros-visit1-aa1510' in l:
                    print(l)
                lines += 1
                # print(line)
            total += lines
            print(dataset, d, lines)
    print(dataset, total)

root = './erda2/sleep-data/resources/processed'
# datasets = ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sof', 'shhs', 'sedf_sc', 'sedf_st']
datasets = ['mros']
for dataset in datasets:
    count_samples(root, dataset)
