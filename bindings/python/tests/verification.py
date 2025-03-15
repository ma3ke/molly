import time
from sys import stdout

import MDAnalysis as MDA
import molly
import numpy as np

END = "\r" if stdout.isatty() else "\n"


def setup_readers(path):
    mda_reader = MDA.coordinates.XTC.XTCReader(
        path, convert_units=False, refresh_offsets=True
    )
    molly_reader = molly.XTCReader(path)

    return mda_reader, molly_reader


def read_all(path):
    """Read and verify all frames, frame-by-frame."""

    mda_reader, molly_reader = setup_readers(path)

    mda_frames = []
    molly_frames = []

    for i in range(mda_reader.n_frames):
        start = time.time()
        mda_positions = mda_reader.trajectory[i].positions
        if i % 10 == 0:
            print(f"{i:>5} mda: {(time.time() - start) * 1000:6.03f} ms/frame", end="")

        start = time.time()
        molly_positions = molly_reader.pop_frame().positions
        if i % 10 == 0:
            print(
                f"     molly: {(time.time() - start) * 1000:6.03f} ms/frame", end="\r"
            )

        assert mda_positions.tolist() == molly_positions.tolist()
        mda_frames.append(mda_positions.copy())
        molly_frames.append(molly_positions.copy())

    return mda_frames, molly_frames


def test_read_frames(path, full_mda_frames, frame_selection, atom_selection) -> float:
    _, molly_reader = setup_readers(path)
    mda_frames = full_mda_frames[frame_selection]
    nframes = len(mda_frames)
    natoms = len(mda_frames[0])
    print(f"\t\t{nframes = }, {natoms = }")
    start = time.time()
    molly_frames = molly_reader.read_frames(
        frame_selection=frame_selection, atom_selection=atom_selection
    )
    duration = time.time() - start

    for i, (mda_positions, molly_frame) in enumerate(zip(mda_frames, molly_frames)):
        molly_positions = molly_frame.positions
        print("\t\t", i, end="\r")

        assert (
            mda_positions.tolist() == molly_positions.tolist()
        ), f"""mda_positions and molly_positions do not match
{mda_positions = }
{molly_positions = }"""

    return duration


def test_read_into_array(
    path, full_mda_frames, frame_selection, atom_selection
) -> float:
    _, molly_reader = setup_readers(path)
    mda_frames = full_mda_frames[frame_selection]
    nframes = len(mda_frames)
    natoms = len(mda_frames[0])
    print(f"\t\t{nframes = }, {natoms = }")
    molly_frames = np.zeros((nframes, natoms, 3), dtype=np.float32)
    molly_boxvecs = np.zeros((nframes, 3, 3), dtype=np.float32)
    start = time.time()
    molly_reader.read_into_array(
        molly_frames,
        molly_boxvecs,
        frame_selection=frame_selection,
        atom_selection=atom_selection,
    )
    duration = time.time() - start

    for i, (mda_positions, molly_positions) in enumerate(zip(mda_frames, molly_frames)):
        print("\t\t", i, end="\r")

        assert (
            mda_positions.tolist() == molly_positions.tolist()
        ), f"{mda_positions = }\n{molly_positions = }"

    return duration


def read_test(path, full_mda_frames, frame_selection=None, atom_selection=None):
    print(f"TEST: {frame_selection = }, {atom_selection = }, {path = }")
    print(" -\tread_frames")
    dur = test_read_frames(path, full_mda_frames, frame_selection, atom_selection)
    print(f"\tOK!\t\tReading took {dur:8.3} s.")
    print(" -\tread_into_array")
    dur = test_read_into_array(path, full_mda_frames, frame_selection, atom_selection)
    print(f"\tOK!\t\tReading took {dur:8.3} s.")


path = "../../tests/trajectories/trajectory_smol.xtc"
full_mda_frames, _ = read_all(path)

# Frame slices.
read_test(path, full_mda_frames, slice(None, None))
read_test(path, full_mda_frames, slice(None, 20))
read_test(path, full_mda_frames, slice(25, 50))
read_test(path, full_mda_frames, slice(None, None, 2))
read_test(path, full_mda_frames, slice(None, 20, 2))
read_test(path, full_mda_frames, slice(25, 50, 2))
read_test(path, full_mda_frames, slice(None, None, 3))
read_test(path, full_mda_frames, slice(None, 20, 3))
read_test(path, full_mda_frames, slice(25, 50, 3))
