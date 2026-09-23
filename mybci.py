from pathlib import Path
import numpy as np
import mne
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True,
        help="Directory of input .edf files. Must be of structure:\n"
             "<dir>\n"
             "    |-Patient 1 subdir\n"
             "    |                |-...R01.edf\n"
             "    |                |-...R02.edf\n"
             "    |                |....\n"
             "    |                |-...R<last-number>.edf\n"
             "    |-Patient 2 subdir\n"
             "    |...\n"
             "IMPORTANT: This program assumes the following order of "
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
    print(list(args.input_dir.iterdir()))


if (__name__=="__main__"):
    main()
