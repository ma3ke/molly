import tempfile
import time
import os
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

    assert molly_reader.pop_frame() is None
    molly_reader.close()
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


def test_write_roundtrip(path):
    """Test writing frames and reading them back."""
    print(f"TEST: write roundtrip, {path = }")

    # Read original frames.
    reader = molly.XTCReader(path)
    original_frames = reader.read_frames()
    nframes = len(original_frames)
    print(f"\tRead {nframes} frames from original file")

    # Write to temp file.
    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as tmp:
        tmp_path = tmp.name

    writer = molly.XTCWriter(tmp_path)
    start = time.time()
    for frame in original_frames:
        writer.write_frame(frame)
    writer.close()
    write_duration = time.time() - start
    print(f"\tWrote {nframes} frames in {write_duration:.3f} s")

    # Read back and verify.
    reader2 = molly.XTCReader(tmp_path)
    roundtrip_frames = reader2.read_frames()

    assert len(roundtrip_frames) == len(original_frames), \
        f"Frame count mismatch: {len(roundtrip_frames)} != {len(original_frames)}"

    for i, (orig, rt) in enumerate(zip(original_frames, roundtrip_frames)):
        assert orig.step == rt.step, f"Frame {i}: step mismatch"
        assert orig.time == rt.time, f"Frame {i}: time mismatch"
        assert orig.precision == rt.precision, f"Frame {i}: precision mismatch"
        assert np.allclose(orig.box, rt.box), f"Frame {i}: box mismatch"
        assert np.allclose(orig.positions, rt.positions, atol=1e-5), \
            f"Frame {i}: positions mismatch"

    # Clean up.
    import os
    os.unlink(tmp_path)

    print(f"\tOK!\t\tRoundtrip verified for {nframes} frames.")


def test_write_from_scratch():
    """Test writing frames and reading them back, but from scratch."""
    print("TEST: write from scratch")
    # Write temp file.
    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as tmp:
        tmp_path = tmp.name

    start = time.time()
    writer = molly.XTCWriter(tmp_path)
    generator = np.random.default_rng(seed=42)
    n_atoms = 1000
    n_frames = 1000
    positions = generator.random((n_frames, n_atoms, 3), dtype=np.float32) * 5.
    box = np.array([
        [5., 0., 0.],
        [0., 5., 0.],
        [0., 0., 5.]
    ], dtype=np.float32)
    for i, frame in enumerate(positions):
        f = molly.Frame()
        f.box = box
        f.positions = frame.flatten()
        f.time = i * 0.02
        f.step = i
        # Precision intentionally left blank.
        writer.write_frame(f)
    writer.close()
    print(f"\tWrote {n_frames} frames {n_atoms} atoms in {time.time() - start:.3f} s")

    # Read back and check if it's the same.
    start = time.time()
    reader = molly.XTCReader(tmp_path)
    read_positions = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    for i in range(n_frames):
        read_positions[i, :, :] = reader.pop_frame().positions
    assert reader.pop_frame() is None

    for i in range(n_frames):
        assert np.allclose(
            positions[i],
            read_positions[i],
            rtol=0.,
            atol=1e-3
        ), f"Frame {i}: Coordinate mismatch."
    print(f"\tRead {n_frames} frames {n_atoms} atoms in {time.time() - start:.3f} s")

    # Clean up.
    import os
    os.unlink(tmp_path)

    print(f"\tOK!\t\tWrite from scratch for {n_frames} frames.")

def test_reference_semantics(path):
    """Tests whether XTCReader's buffered frame has reference semantics."""
    print("TEST: XTCReader.frame reference semantics")
    reader = molly.XTCReader(path)
    # Read 1 frame.
    f = reader.pop_frame()
    # Overwrite it through the reference.
    # (The first frame in this particular file is at MD step 0.)
    assert f.step == 0
    f.step = 5
    # `reader.frame` and anything returned by pop_frame() and read_frame() should be a reference.
    assert reader.frame.step == 5
    # Read another frame.
    reader.read_frame()
    # `f` should remain a frame containing the old positions.
    assert f.step == 5
    f.step = 0
    # `reader.frame` should be a new frame.
    assert reader.frame.step > 0
    print("\tOK!")
    print("\tXTCReader.frame has reference semantics")


def test_append():
    """Test appending to existing .xtc files."""
    print("TEST: append")
    # Write temp file.
    with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as tmp:
        tmp_path = tmp.name
    start = time.time()
    writer = molly.XTCWriter(tmp_path)
    generator = np.random.default_rng(seed=1234)
    n_atoms = 1000
    n_frames = 1000
    append_every = 100

    positions = generator.random((n_frames, n_atoms, 3), dtype=np.float32) * 5.
    box = np.array([
        [5., 0., 0.],
        [0., 5., 0.],
        [0., 0., 5.]
    ], dtype=np.float32)
    for i, frame in enumerate(positions):
        f = molly.Frame()
        f.box = box
        f.positions = frame.flatten()
        f.time = i * 0.02
        f.step = i
        # Precision intentionally left blank.
        writer.write_frame(f)
        if i > 0 and i % append_every == 0:
            writer.close()
            writer = molly.XTCWriter(tmp_path, append=True)
    writer.close()
    print(f"\tWrote {n_frames} frames {n_atoms} atoms in {time.time() - start:.3f} s")

    # Read back and check if it's the same.
    start = time.time()
    reader = molly.XTCReader(tmp_path)
    read_positions = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    for i in range(n_frames):
        read_positions[i, :, :] = reader.pop_frame().positions
    assert reader.pop_frame() is None

    for i in range(n_frames):
        assert np.allclose(
            positions[i],
            read_positions[i],
            rtol=0.,
            atol=1e-3
        ), f"Frame {i}: Coordinate mismatch."
    print(f"\tRead {n_frames} frames {n_atoms} atoms in {time.time() - start:.3f} s")

    # Clean up.
    import os
    os.unlink(tmp_path)

    print(f"\tOK!\tAppend for {n_frames} frames.")


def test_skipping_frames(path):
    """Read some frames and verify them, then skip some frames."""

    print("TEST: skip frames")
    mda_reader, molly_reader = setup_readers(path)

    for i in range(mda_reader.n_frames):
        if i % 5 > 0:
            molly_reader.skip_frame()
        else:
            mda_positions = mda_reader.trajectory[i].positions
            molly_positions = molly_reader.pop_frame().positions
            assert mda_positions.tolist() == molly_positions.tolist()
    print(f"\tOK!\tSkip frames for {mda_reader.n_frames} frames.")

def test_skip_and_tell(path):
    """Compare the file positions given by tell() and skip_frame()."""

    print("TEST: skip and tell")
    offsets = []
    reader = molly.XTCReader(path)

    while (offset := reader.skip_frame()) is not None:
        offsets.append(offset)
    
    reader.close()
    reader = molly.XTCReader(path)
    for offset in offsets:
        reader.pop_frame()
        assert offset == reader.tell()

    reader.close()
    print("\tOK!\tSkip and tell.")

def test_read_iter(path):
    """Compare the method .pop_frame() with the iterator .frames() and method .read_frames()."""

    print("TEST: read iter")
    reader = molly.XTCReader(path)

    frames_pop = []
    while (f := reader.pop_frame()):
        frames_pop.append(f)

    reader.home()
    frames_iter = list(reader.frames())

    reader.home()
    frames_read = reader.read_frames()

    assert len(frames_pop) == len(frames_iter) == len(frames_read), (
        f"len pop: {len(frames_pop)}\n"
        + f"len iter: {len(frames_iter)}\n"
        + f"len read: {len(frames_read)}"
    )
    for f1, f2, f3 in zip(frames_pop, frames_iter, frames_read):
        assert np.allclose(f1.positions, f2.positions)
        assert np.allclose(f1.positions, f3.positions)
        assert np.allclose(f1.box, f2.box)
        assert np.allclose(f1.box, f3.box)
        assert np.isclose(f1.time, f2.time)
        assert np.isclose(f1.time, f2.time)
        assert f1.step == f2.step == f3.step

    reader.close()
    print("\tOK!\tRead iter.")

if __name__ == "__main__":
    # Chdir to /bindings/python/tests.
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    path = "../../../tests/trajectories/trajectory_smol.xtc"
    full_mda_frames, _ = read_all(path)

    # Reader tests, with different frame slices.
    read_test(path, full_mda_frames, slice(None, None))
    read_test(path, full_mda_frames, slice(None, 20))
    read_test(path, full_mda_frames, slice(25, 50))
    read_test(path, full_mda_frames, slice(None, None, 2))
    read_test(path, full_mda_frames, slice(None, 20, 2))
    read_test(path, full_mda_frames, slice(25, 50, 2))
    read_test(path, full_mda_frames, slice(None, None, 3))
    read_test(path, full_mda_frames, slice(None, 20, 3))
    read_test(path, full_mda_frames, slice(25, 50, 3))

    # Reader tests, additional QoL methods.
    test_skipping_frames(path)
    test_skip_and_tell(path)
    test_read_iter(path)

    # Reader tests, semantics.
    test_reference_semantics(path)

    # Writer tests.
    test_write_roundtrip(path)
    test_write_from_scratch()
    test_append()
