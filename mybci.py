from pathlib import Path
import numpy as np
import mne
import argparse
import re

from utils.mne_stuff import find_channel
from utils.fourier import compute_spectrogram_stft


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


def my_add_element(to_what: dict, where: str, what: np.ndarray):
    if (where not in to_what.keys()):
        to_what[where] = [what]
    else:
        to_what[where].append(what)

def main():
    args = parse_args()
    all_patient_paths = list(args.input_dir.iterdir())
    tagged_entries = {}
    for patient_path in all_patient_paths:
        if (patient_path.is_dir()):
            print(f"patient: {patient_path.name}")
            local_file_paths = list(patient_path.iterdir())
            avg_power_per_freq = [[], []]
            for local_file_path in local_file_paths:
                res = RUN_RE.search(local_file_path.name)
                if (res is not None):
                    raw = mne.io.read_raw_edf(
                        local_file_path,
                        preload=True,
                        verbose="ERROR"
                    )
                    exp_id = int(res.group(1))
                    channel_name = find_channel(raw, "Cz..")
                    channel_i = raw.ch_names.index(channel_name)
                    voltage = raw.get_data()[channel_i]
                    sfreq = raw.info["sfreq"]
#                    print(f"experiment id: {exp_id}, "
#                          f"freq: {sfreq}"
#                          f"shape: {voltage.shape}")
                    freqs, times, power_db = compute_spectrogram_stft(
                        voltage,
                        sfreq,
                        window_seconds=1.0,
                        overlap_ratio=0.9,
                    )
                    if (exp_id == 1):
                        # collect eyes open background here
                        avg_power_per_freq[0] = power_db.mean(axis=-1)
                    elif (exp_id == 2):
                        # collect eyes closed background here
                        avg_power_per_freq[1] = power_db.mean(axis=-1)
                    else:
                        for on, dur, des in zip(
                                raw.annotations.onset,
                                raw.annotations.duration,
                                raw.annotations.description):
                            if (des == "T0"):
                                continue
                            start = on
                            end = on + dur
                            start_i = np.searchsorted(times, start)
                            end_i = np.searchsorted(times, end)
                            image = power_db[:, start_i:end_i].copy()
                            if (image.shape[1] != 41):
                                continue
                            loc = patient_path.name
                            if (exp_id in [3, 7, 11, 5, 9, 13]):
                                loc += "_o_"
                            else:
                                loc += "_c_"
                            if (des == "T1"):
                                if (exp_id in [3, 7, 11, 4, 6, 12]):
                                    loc += "lh"
                                else:
                                    loc += "2h"
                            else:
                                if (exp_id in [3, 7, 11, 4, 6, 12]):
                                    loc += "rh"
                                else:
                                    loc += "ft"
                            my_add_element(tagged_entries, loc, image)
#            print("finally, avg ppf bg for open eyes:")
#            print(avg_power_per_freq[0])
#            print("and closed:")
#            print(avg_power_per_freq[1])
    print(tagged_entries)


if (__name__=="__main__"):
    main()
