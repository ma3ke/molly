mod common;
use common::trajectories;

// The trajectory for this test is very large so it does not live within the repository. If the
// trajectory is not present, we always pass the test. Discussed in #20.
#[test]
fn read_large_steps() {
    let Ok(mut molly_reader) = molly::XTCReader::open(trajectories::LARGE_STEP) else {
        eprint!(
            "WARNING: Skipping test at {}:{} because the test data is not present.",
            file!(),
            line!()
        );
        return;
    };

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

#[test]
fn test_skip_positions() {
    let mut reader = molly::XTCReader::open(trajectories::COB).unwrap();
    let mut frame = molly::Frame::default();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    let step2 = frame.step;

    reader.home().unwrap();

    let header1 = reader.read_header().unwrap();
    reader.skip_positions(&header1).unwrap();
    let header2 = reader.read_header().unwrap();
    assert_eq!(step2, header2.step);
}

#[test]
fn test_skip_to_time() {
    let mut reader = molly::XTCReader::open(trajectories::ADK).unwrap();
    let mut frame = molly::Frame::default();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    let step3 = frame.step;
    let time3 = frame.time;
    println!("{step3} {time3}");

    reader.home().unwrap();

    let header = reader.skip_to_time(time3).unwrap();
    assert_eq!(time3, header.time);
}

#[test]
fn test_skip_frames() {
    let mut reader = molly::XTCReader::open(trajectories::ADK).unwrap();
    let mut frame = molly::Frame::default();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    let step3 = frame.step;
    let time3 = frame.time;
    println!("{step3} {time3}");

    reader.home().unwrap();

    reader.skip_frames(2).unwrap();
    let header = reader.read_header().unwrap();
    assert_eq!(time3, header.time);
}

#[test]
fn test_seek_prev() {
    let mut reader = molly::XTCReader::open(trajectories::ADK).unwrap();
    let mut frame = molly::Frame::default();
    reader.read_frame(&mut frame).unwrap();
    reader.read_frame(&mut frame).unwrap();
    let step2 = frame.step;
    let time2 = frame.time;
    println!("{step2} {time2}");

    let header = reader.seek_prev().unwrap();
    assert_eq!(time2, header.time);
}

#[test]
fn test_read_last() {
    let mut reader = molly::XTCReader::open(trajectories::ADK).unwrap();
    let frames = reader.read_all_frames().unwrap();
    let last_time = frames.last().unwrap().time;

    reader.home().unwrap();

    let mut fr = molly::Frame::default();
    reader.read_last_frame(&mut fr).unwrap();
    assert_eq!(last_time, fr.time);
}
