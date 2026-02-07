use xdrfile::Trajectory;

mod common;
use common::trajectories;

fn compare(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let mut molly_reader = molly::XTCReader::open(&path)?;
    let mut xdr_reader =
        xdrfile::XTCTrajectory::open_read(&path).expect("couldn't open file using xdrfile");

    let mut molly_frame = molly::Frame::default();
    let num_atoms = xdr_reader
        .get_num_atoms()
        .expect("couldn't get number of atoms from xdrfile");
    let mut xdr_frame = xdrfile::Frame::with_len(num_atoms);

    // Compare our buffered and unbuffered implementations.
    let mut buffered_read_frames = Vec::new();
    molly_reader.read_frames::<true>(
        &mut buffered_read_frames,
        &molly::selection::FrameSelection::All,
        &molly::selection::AtomSelection::All,
    )?;
    molly_reader.home()?;
    let mut unbuffered_read_frames = Vec::new();
    molly_reader.read_frames::<false>(
        &mut unbuffered_read_frames,
        &molly::selection::FrameSelection::All,
        &molly::selection::AtomSelection::All,
    )?;
    molly_reader.home()?;
    assert_eq!(
        buffered_read_frames, unbuffered_read_frames,
        "the buffered and unbuffered readers should give identical results"
    );

    // Compare against other implementations.
    while molly_reader.read_frame(&mut molly_frame).is_ok() {
        xdr_reader
            .read(&mut xdr_frame)
            .expect("couldn't read xdrfile frame");

        let molly_positions = molly_frame.coords().collect::<Vec<_>>();
        let xdr_positions = xdr_frame.coords.as_slice();

        assert_eq!(
            xdr_frame.step, molly_frame.step as usize,
            "molly and xdrfile read a different simulation step"
        );
        assert_eq!(
            molly_positions.len(),
            xdr_positions.len(),
            "molly and xdrfile read a different number of atoms"
        );

        for i in 0..molly_positions.len() {
            let molly_pos = molly_positions[i];
            let xdr_pos = xdr_positions[i];

            assert_eq!(
                molly_pos, xdr_pos,
                "position {i} for molly and xdrfile does not match"
            );
        }
    }

    // Make sure that after molly_reader is done, xdr_reader is also done.
    assert!(
        molly_reader.read_frame(&mut molly_frame).is_err(),
        "idiot check, molly reader should be done by now"
    );
    assert!(
        xdr_reader.read(&mut xdr_frame).is_err(),
        "xdrfile reader should be done by now"
    );

    Ok(())
}

#[test]
fn compare_adk() -> std::io::Result<()> {
    compare(trajectories::ADK)
}

#[test]
fn compare_aux() -> std::io::Result<()> {
    compare(trajectories::AUX)
}

#[test]
fn compare_cob() -> std::io::Result<()> {
    compare(trajectories::COB)
}

#[test]
fn compare_smol() -> std::io::Result<()> {
    compare(trajectories::SMOL)
}

#[test]
fn compare_ten() -> std::io::Result<()> {
    compare(trajectories::TEN)
}

#[test]
fn compare_xyz() -> std::io::Result<()> {
    compare(trajectories::XYZ)
}

#[test]
fn compare_bad() -> std::io::Result<()> {
    compare(trajectories::BAD)
}

#[test]
fn compare_delinyah() -> std::io::Result<()> {
    compare(trajectories::DELINYAH)
}
