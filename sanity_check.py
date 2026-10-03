# import os

# def count_samples(root, dataset):
#     path = os.path.join(root, dataset, 'views/fixed_split')
#     total = 0
#     for d in ['train', 'val', 'test']:
#         path_ = os.path.join(path, d, "LIST_OF_FILES.txt")
#         with open(path_, 'r') as fp:
#             lines = 0
#             # lines = sum(1 for line in fp)
#             for l in fp.readlines():
#                 if 'mros-visit1-aa1510' in l:
#                     print(l)
#                 lines += 1
#                 # print(line)
#             total += lines
#             print(dataset, d, lines)
#     print(dataset, total)

# root = './erda2/sleep-data/resources/processed'
# # datasets = ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sof', 'shhs', 'sedf_sc', 'sedf_st']
# datasets = ['mros']
# for dataset in datasets:
#     count_samples(root, dataset)

import h5py
from pathlib import Path


def merge_hdf5_external(output_file, input_files):
    """
    Create a merged HDF5 file using external links.

    Parameters
    ----------
    output_file : str
        Path to merged HDF5 file.

    input_files : list[str]
        List of source HDF5 files.
    """

    output_path = Path(output_file)

    with h5py.File(output_path, "w") as fout:

        for input_file in input_files:
            input_path = Path(input_file)

            # Use relative path so merged file is portable
            relative_path = input_path.relative_to(output_path.parent)

            with h5py.File(input_path, "r") as fin:

                for key in fin.keys():

                    if key in fout:
                        raise ValueError(
                            f"Duplicate top-level key '{key}' "
                            f"found in {input_file}"
                        )

                    # Create external link
                    fout[key] = h5py.ExternalLink(
                        str(relative_path),
                        f"/{key}"
                    )


# Example usage
# merge_hdf5_external(
#     "merged.h5",
#     [
#         "file1.h5",
#         "file2.h5",
#         "file3.h5",
#     ]
# )

# import h5py

excluded_samples = {
        'abc': [],
        'ccshs': [],
        'cfs': [],
        'chat': ['chat-baseline-300927'],
        'dcsm': [],
        'homepap': ['1600052', '1600138', '1600280', '1600047', '1600194', '1600361', '1600087', '1600368', '1600203'],
        'mesa': [],
        'mros': ['visit1-aa2180', 'visit1-aa3370', 'visit1-aa1367', 'visit1-aa1715', 'visit1-aa1900', 'visit1-aa3903', 'visit1-aa3411'],
        'phys': [],
        'sedf_sc': [],
        'sedf_st': [],
        'shhs': [],
        'sof': []
    }

remove = []

def preview_h5(name, obj):
    
    if isinstance(obj, h5py.Dataset):
        # print(f"Dataset '{name}': {obj.shape}")
        # for k, vs in excluded_samples.items():
        #     for v in vs:
        #         if k in name and v in name:
        #             print(f"Excluded sample '{name}', dataset")
        pass
    elif isinstance(obj, h5py.Group):
        for k, vs in excluded_samples.items():
            for v in vs:
                if k in name and v in name:
                    # print(f"Excluded sample '{name}', group")
                    if name.count('/') == 2:  # only remove top-level groups, not datasets
                        remove.append(name)

with h5py.File('/home/extra/sleep-data/processed_data.h5', 'r') as f:
    f.visititems(preview_h5)

print(remove)

with h5py.File('/home/extra/sleep-data/processed_data.h5', 'a') as f:
    for name in remove:
        del f[name]
        
def preview_h5_2(name, obj):
    
    if isinstance(obj, h5py.Dataset):
        # print(f"Dataset '{name}': {obj.shape}")
        # for k, vs in excluded_samples.items():
        #     for v in vs:
        #         if k in name and v in name:
        #             print(f"Excluded sample '{name}', dataset")
        pass
    elif isinstance(obj, h5py.Group):
        for k, vs in excluded_samples.items():
            for v in vs:
                if k in name and v in name:
                    print(f"Excluded sample '{name}', group")
                    # remove.append(name)

with h5py.File('/home/extra/sleep-data/processed_data.h5', 'r') as f:
    f.visititems(preview_h5_2)