import shutil
import os

def copy_split(split_dir, out_dir):
    """
    Copy the split directory to a new location. This is useful for creating a copy of the split for training on a different machine or for backup purposes.

    Args:
        split_dir: The directory containing the split to be copied.
        out_dir: The directory where the copied split should be saved.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"Output directory {out_dir} already exists. Files may be overwritten.")
    for item in os.listdir(split_dir):
        s = os.path.join(split_dir, item)
        d = os.path.join(out_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def change_root_in_list_file(list_file_path, new_root, excluded_samples=[]):
    """
    Change the root directory in a list file. This is useful for updating the paths in the list file after copying the split to a new location.

    Args:
        list_file_path: The path to the list file to be updated.
        old_root: The old root directory to be replaced.
        new_root: The new root directory to replace with.
    """
    with open(list_file_path, "r") as f:
        lines = f.readlines()
    with open(list_file_path, "w") as f:
        for line in lines:
            line = line.strip().split('/')[-1]
            exclude = False
            for excluded_sample in excluded_samples:
                # print(line, excluded_sample)
                if excluded_sample in line:
                    print(f"Excluding sample {line} from {list_file_path}")
                    exclude = True
                    break
            if exclude:
                continue
            updated_line = str(os.path.join(new_root, line))
            f.write(updated_line + "\n")
            # updated_line = line.replace(old_root, new_root)
            # f.write(updated_line)

if __name__ == "__main__":
    split_dirs = [
        '/home/jojuo/Documents/UCPH-RA/U-Sleep-Keras-3.0/erda/sleep-data/resources/processed/' + dataset + '/views' for dataset in ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sedf_sc', 'sedf_st', 'shhs', 'sof']
    ]
    
    out_dirs = [
        '/home/jojuo/Documents/UCPH-RA/U-Sleep-Keras-3.0/erda/sleep-data/resources/processed/' + dataset + '/views_local' for dataset in ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sedf_sc', 'sedf_st', 'shhs', 'sof']
    ]
    
    new_roots = [f'/home/jojuo/Documents/UCPH-RA/U-Sleep-Keras-3.0/erda/sleep-data/resources/processed/{dataset}' for dataset in ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sedf_sc', 'sedf_st', 'shhs', 'sof']]
    
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
    
    for split_dir, out_dir, new_root, dataset in zip(split_dirs, out_dirs, new_roots, ['abc', 'ccshs', 'cfs', 'chat', 'dcsm', 'homepap', 'mesa', 'mros', 'phys', 'sedf_sc', 'sedf_st', 'shhs', 'sof']):
        print(f"Copying split from {split_dir} to {out_dir}...")
        copy_split(split_dir, out_dir)
        print(f"Updating list files in {out_dir}...")
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                # print(file)
                if file.endswith(".txt"):
                    list_file_path = os.path.join(root, file)
                    change_root_in_list_file(list_file_path, new_root, excluded_samples[dataset])
        print(f"Finished processing {out_dir}.")
    
