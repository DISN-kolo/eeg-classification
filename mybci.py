from pathlib import Path
import numpy as np
import mne
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-list", nargs="+", type=Path, required=True,
        help="EDF files to read. Corresponding events are assumed to be named"
             " as <file>.event"
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    raw0 = mne.io.read_raw_edf(args.input_list[0])
    print(raw0)

if (__name__=="__main__"):
    main()
