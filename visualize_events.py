"""
Plot .edf waveforms, stacked, T0/1/2 events colored in the background.

(T0 is rest, T1 is the onset of left-fist movement or imagery in unilateral
fist runs, and the both-fists movement or imagery in bilateral hand/foot runs.
T2 is the right-fist movement or imagery in unilateral fist runs,
and the both-feet movement or imagery in bilateral hand/foot runs.)

Each run gets one row showing a representative channel (left y-axis) and
the RMS across all 64 channels (right y-axis).
"""
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import mne
import numpy as np


# from .venv/lib/python3.13/site-packages/mne/datasets/eegbci/eegbci.py:
#
# run        task
# 1          Baseline, eyes open
# 2          Baseline, eyes closed
# 3, 7, 11   Motor execution: left vs right hand
# 4, 8, 12   Motor imagery: left vs right hand
# 5, 9, 13   Motor execution: hands vs feet
# 6, 10, 14  Motor imagery: hands vs feet
TASK_RUN_MATCH = {
    "Baseline, eyes open": [1],
    "Baseline, eyes closed": [2],
    "Motor execution: T1 = left hand, T2 = right hand": [3, 7, 11],
    "Motor imagery: T1 = left hand, T2 = right hand": [4, 8, 12],
    "Motor execution: T1 = hands, T2 = feet": [5, 9, 13],
    "Motor imagery: T1 = hands, T2 = feet": [6, 10, 14],
}

EVENT_COLORS = {
    "T0": "#d0d0d0",
    "T1": "#a0a0f0",
    "T2": "#ff9090",
}

LINE_COLORS = [
    "#2d0000",
    "#002d00",
    "#00002d",
    "#1d1d00",
    "#001d1d",
    "#1d001d",
]

MY_RMS = "#10a020"

FILENAME_RE = re.compile(r"S(\d+)R(\d+)\.edf$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-list", nargs="+", type=Path, required=True,
        help="EDF files to plot. 1 row = 1 file. Sorted by run number"
             " before plotting."
    )
    parser.add_argument(
        "-c", "--channel", default="Cz",
        help="Channel to plot on the left axis"
             " (default: Cz)."
    )
    parser.add_argument(
        "-a", "--all-channels", action="store_true",
        help="All channels mode for a singular input. Warning: lag"
    )
    args = parser.parse_args()
    if (len(args.input_list) != 1 and args.all_channels):
        parser.error("All channels moode only supports a singular input")

    return args


def run_number(path):
    match = FILENAME_RE.search(path.name)
    if match is None:
        return None
    else:
        return int(match.group(2))


def find_channel(raw, name):
    for ch_name in raw.ch_names:
        if (ch_name.rstrip(".") == name or ch_name == name):
            return ch_name

    raise ValueError(f"channel {name!r} not found in {raw.ch_names}")


def run_title(path):
    run = run_number(path)
    if (run is None):
        return path.name

    for run_name, run_nums in zip(
            TASK_RUN_MATCH.keys(),
            TASK_RUN_MATCH.values()):
        if (run in run_nums):
            return f"{path.name}: {run_name}"

    return f"{path.name}: unknown run"


def plot_event_spans(ax, raw):
    for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description):
        color = EVENT_COLORS.get(description, "grey")
        ax.axvspan(
            onset, onset + duration,
            color=color, alpha=1.0, zorder=0, linewidth=0,
        )


def plot_run(ax, path, channel_name):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    data_uv = raw.get_data() * 1e6
    times = raw.times

    plot_event_spans(ax, raw)

    ch_name = find_channel(raw, channel_name)
    ch_idx = raw.ch_names.index(ch_name)
    ax.plot(times, data_uv[ch_idx], color="#101010", linewidth=0.6)
    ax.set_ylabel(f"{channel_name} (uV)", fontsize=16)

    rms_uv = (data_uv ** 2).mean(axis=0) ** 0.5
    twin = ax.twinx()
    twin.plot(times, rms_uv, color=MY_RMS, linewidth=0.6, alpha=0.8)
    twin.set_ylabel("RMS (uV)", fontsize=16, color=MY_RMS)

    ax.set_title(run_title(path), font="monospace", fontsize=16, loc="left")


def plot_montage(ax, path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    data_uv = raw.get_data() * 1e6
    times = raw.times
    n_channels = len(raw.ch_names)

    plot_event_spans(ax, raw)

    spacing = 4.0 * data_uv.std()
    offsets = spacing * np.arange(n_channels)

    for i in range(n_channels - 1):
        ax.axhline(
            offsets[i] + spacing / 2,
            color="#999999",
            linewidth=1,
            zorder=1,
        )

    for offset in offsets:
        ax.axhline(
            offset,
            color="#999999",
            linewidth=0.5,
            linestyle=":",
            zorder=1,
        )

    for ch_idx in range(n_channels):
        ax.plot(
            times,
            data_uv[ch_idx] + offsets[ch_idx],
            color=LINE_COLORS[ch_idx % len(LINE_COLORS)],
            linewidth=0.4,
            zorder=2,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(
        [name.rstrip(".") for name in raw.ch_names], fontsize=7,
    )
    ax.set_ylim(offsets[0] - spacing, offsets[-1] + spacing)
    ax.invert_yaxis()
    ax.set_title(run_title(path), font="monospace", fontsize=14, loc="left")


def main():
    args = parse_args()
    paths = sorted(
        args.input_list,
        key=lambda p: (run_number(p) is None, run_number(p) or 0, p.name),
    )

    if (not args.all_channels):
        fig, axes = plt.subplots(
            len(paths), 1,
            figsize=(14, 2.2 * len(paths)),
            squeeze=False,
            layout="constrained",
        )
        for ax_row, path in zip(axes[:, 0], paths):
            plot_run(ax_row, path, args.channel)
        axes[-1, 0].set_xlabel("time (s)", fontsize=16)
    else:
        fig, ax = plt.subplots(figsize=(14, 12), layout="constrained")
        plot_montage(ax, paths[0])
        ax.set_xlabel("time (s)", fontsize=16)

    fig.suptitle("Events: T0 is grey, T1 is blue, T2 is red", fontsize=22)
    plt.show()


if (__name__ == "__main__"):
    main()
