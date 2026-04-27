# from psg_utils.downloads.phys.phys import preprocess_phys_hypnograms
import logging
import os
from glob import glob

# 
def preprocess_phys_hypnograms(dataset_folder_path):
    """
    Preprocesses files from the PHYS dataset.
    OBS: Only processes the hypnogram (.arousal) files
         Creates 1 new file in each PHYS subject dir (.ids format)

    :param dataset_folder_path: path to PHYS file on local disk
    :return: None
    """
    import numpy as np
    from wfdb.io import rdann
    from psg_utils.io.file_writers import to_ids
    from psg_utils.io.high_level_file_loaders import load_psg
    from psg_utils.hypnogram import SparseHypnogram
    from psg_utils import Defaults

    # Get list of subject folders
    subject_folders = glob(os.path.join(dataset_folder_path, "tr*"))
    LABEL_MAP = {
        'N1': "N1",
        'N2': "N2",
        'N3': "N3",
        'R': "REM",
        'W': "W",
    }

    for i, folder in enumerate(subject_folders):
        name = os.path.split(os.path.abspath(folder))[-1]
        print(f"{i+1}/{len(subject_folders)}", name)

        # Get sleep-stages
        edf_file = folder + f"/{name}.mat"
        org_hyp_file = folder + f"/{name}.arousal"
        new_hyp_file = folder + f"/{name}.arousal.st"
        out_path = new_hyp_file.replace(".arousal.st", "-HYP.ids")
        if os.path.exists(out_path):
            print("Exists, skipping...")
            continue
        if os.path.exists(org_hyp_file):
            os.rename(org_hyp_file, new_hyp_file)

        psg, header = load_psg(edf_file, load_channels=['C3-M2'])
        hyp = rdann(new_hyp_file[:-3], "st")

        sample_rate = header["sample_rate"]
        psg_length_sec = len(psg)/sample_rate

        pairs = zip(hyp.aux_note, hyp.sample)
        stages = [s for s in pairs if not ("(" in s[0] or ")" in s[0])]
        stages = [(s[0], int(s[1]/sample_rate)) for s in stages]
        stages, starts = map(list, zip(*stages))
        stages = [LABEL_MAP[s] for s in stages]

        if starts[0] != 0:
            i = [0] + starts
            s = ["UNKNOWN"] + stages
            print('append to ', starts[0])
        else:
            i, s = starts, stages
        diff = psg_length_sec - i[-1]
        assert diff >= 0
        if diff // 30 * 30 == 0:
            print('trail: ', diff)
        # print(diff // 30 * 30)
        # d = list(np.diff(i)) + [(diff//30) * 30]
        d = list(np.diff(i)) + [int(diff)]
        
        # print(i, '\n', d, sample_rate)
        # print(len(s), len(d))
        # print(s[-1], d[-1])
        # print()
        # print(len(d))
        # print(len(s))
        SparseHypnogram(i, d, [Defaults.get_stage_string_to_class_int()[s_] for s_ in s], 30)
        to_ids(i, d, s, out_path)
preprocess_phys_hypnograms('./erda2/sleep-data/resources/phys')