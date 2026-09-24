from pathlib import Path
import numpy as np
import mne
import argparse
import re


RUN_RE = re.compile(r"(\d+).edf$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True,
        help="Directory of input .edf files. Must contain subfolders "
             "corresponding to persons, which in turn must contain "
             "the files named as '[anything]R<00..14>.edf'. "
             "This is because the program assumes the following order of "
             "recordings - R01 is baseline with eyes open, R02 is baseline "
             "with eyes closed, while R03..R06 are eyes open/closed alterna"
             "ting with the first two being left/right hand movement (T1/T2) "
             "and the last two being hands/feet movement (T1/T2), repeated "
             "afterwards (in the original dataset repeated 2 more times "
             "until R14)"
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    all_patient_paths = list(args.input_dir.iterdir())
    for patient_path in all_patient_paths:
        if (patient_path.is_dir()):
            print(f"patient: {patient_path.name}")
            local_file_paths = list(patient_path.iterdir())
            for local_file_path in local_file_paths:
                res = RUN_RE.search(local_file_path.name)
                if (res is not None):
                    exp_id = int(res.group(1))
                    raw = mne.io.read_raw_edf(
                        local_file_path,
                        preload=True,
                        verbose="ERROR"
                    )
                    voltage = raw.get_data()
                    print(f"experiment id: {exp_id}, "
                          f"freq: {raw.info['sfreq']}, "
                          f"shape: {voltage.shape}")



if (__name__=="__main__"):
    main()
