mod common;
use common::trajectories;

#[test]
fn read_large_steps() {
    let mut molly_reader = molly::XTCReader::open(trajectories::LARGE_STEP).unwrap();

    let mut buffered_read_frames = Vec::new();
    molly_reader
        .read_frames::<true>(
            &mut buffered_read_frames,
            &molly::selection::FrameSelection::All,
            &molly::selection::AtomSelection::All,
        )
        .expect("Could not read frames using buffered reading.");
    molly_reader
        .home()
        .expect("Could not reset reader to initial position.");

    let mut unbuffered_read_frames = Vec::new();
    molly_reader
        .read_frames::<false>(
            &mut unbuffered_read_frames,
            &molly::selection::FrameSelection::All,
            &molly::selection::AtomSelection::All,
        )
        .expect("Could not read frames using unbuffered reading.");
    molly_reader
        .home()
        .expect("Could not reset reader to initial position.");
    assert_eq!(
        buffered_read_frames, unbuffered_read_frames,
        "the buffered and unbuffered readers should give identical results"
    );
    assert_eq!(buffered_read_frames.len(), 11);
    assert_eq!(unbuffered_read_frames.len(), 11);

    let expected_steps = [
        3000000000, 3000005000, 3000010000, 3000015000, 3000020000, 3000025000, 3000030000,
        3000035000, 3000040000, 3000045000, 3000050000,
    ]
    .into_iter();

    assert!(buffered_read_frames
        .iter()
        .map(|frame| frame.step)
        .eq(expected_steps.clone()));

    assert!(unbuffered_read_frames
        .iter()
        .map(|frame| frame.step)
        .eq(expected_steps));
}
