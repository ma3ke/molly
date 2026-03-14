mod common;

use std::io::{self, Cursor};

use common::trajectories;
use molly::{Frame, XTCReader, XTCWriter};

fn assert_frames_eq(original: &[Frame], roundtrip: &[Frame]) {
    assert_eq!(
        original.len(),
        roundtrip.len(),
        "frame count mismatch: original {} vs roundtrip {}",
        original.len(),
        roundtrip.len()
    );

    for (i, (orig, rt)) in original.iter().zip(roundtrip.iter()).enumerate() {
        assert_eq!(orig.step, rt.step, "frame {i}: step mismatch");
        assert_eq!(orig.time, rt.time, "frame {i}: time mismatch");
        assert_eq!(orig.boxvec, rt.boxvec, "frame {i}: boxvec mismatch");
        assert_eq!(
            orig.positions.len(),
            rt.positions.len(),
            "frame {i}: position count mismatch"
        );

        for (j, (op, rp)) in orig.positions.iter().zip(rt.positions.iter()).enumerate() {
            let diff = (op - rp).abs();
            assert!(
                diff < 1e-5,
                "frame {i}, position {j}: value mismatch (original {op}, roundtrip {rp}, diff {diff})"
            );
        }
    }
}

macro_rules! roundtrip_test {
    ($name:ident, $path:expr) => {
        #[test]
        fn $name() -> io::Result<()> {
            let frames = XTCReader::open($path)?.read_all_frames()?;
            let mut buf = Vec::new();
            {
                let mut writer = XTCWriter::new(Cursor::new(&mut buf));
                for frame in frames.iter() {
                    writer.write_frame(frame)?;
                }
            }
            let roundtrip = XTCReader::new(Cursor::new(&buf)).read_all_frames()?;
            assert_frames_eq(&frames, &roundtrip);
            Ok(())
        }
    };
}

roundtrip_test!(roundtrip_adk, trajectories::ADK);
roundtrip_test!(roundtrip_aux, trajectories::AUX);
roundtrip_test!(roundtrip_cob, trajectories::COB);
roundtrip_test!(roundtrip_smol, trajectories::SMOL);
roundtrip_test!(roundtrip_ten, trajectories::TEN);
roundtrip_test!(roundtrip_xyz, trajectories::XYZ);
roundtrip_test!(roundtrip_delinyah, trajectories::DELINYAH);

fn encode_size(path: &str) -> io::Result<usize> {
    let frames = XTCReader::open(path)?.read_all_frames()?;
    let mut buf = Vec::new();
    {
        let mut writer = XTCWriter::new(std::io::Cursor::new(&mut buf));
        for frame in frames.iter() {
            writer.write_frame(frame)?;
        }
    }
    Ok(buf.len())
}

#[test]
fn write_frame_parts_noncontiguous_ten() -> io::Result<()> {
    let frames = XTCReader::open(trajectories::TEN)?.read_all_frames()?;
    let atom_indices: &[usize] = &[1, 3, 7];

    let mut buf = Vec::new();
    {
        let mut writer = XTCWriter::new(Cursor::new(&mut buf));
        for frame in &frames {
            let coords: Vec<[f32; 3]> = atom_indices
                .iter()
                .map(|&i| frame.positions[3 * i..3 * i + 3].try_into().unwrap())
                .collect();
            writer.write_frame_parts(frame.step, frame.time, frame.boxvec, coords.iter(), frame.precision)?;
        }
    }

    let roundtrip = XTCReader::new(Cursor::new(&buf)).read_all_frames()?;

    assert_eq!(frames.len(), roundtrip.len());
    for (i, (orig, rt)) in frames.iter().zip(roundtrip.iter()).enumerate() {
        assert_eq!(orig.step, rt.step, "frame {i}: step mismatch");
        assert_eq!(orig.time, rt.time, "frame {i}: time mismatch");
        assert_eq!(orig.boxvec, rt.boxvec, "frame {i}: boxvec mismatch");
        assert_eq!(rt.positions.len(), atom_indices.len() * 3, "frame {i}: position count mismatch");
        for (j, &atom_idx) in atom_indices.iter().enumerate() {
            for k in 0..3 {
                let orig_val = orig.positions[3 * atom_idx + k];
                let rt_val = rt.positions[3 * j + k];
                let diff = (orig_val - rt_val).abs();
                assert!(
                    diff < 1e-5,
                    "frame {i}, atom {atom_idx}, coord {k}: mismatch ({orig_val} vs {rt_val}, diff {diff})"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn write_frame_parts_noncontiguous_delinyah() -> io::Result<()> {
    let frames = XTCReader::open(trajectories::DELINYAH)?.read_all_frames()?;
    let natoms = frames[0].positions.len() / 3;
    let atom_indices: Vec<usize> = (3..natoms).step_by(7).collect();

    let mut buf = Vec::new();
    {
        let mut writer = XTCWriter::new(Cursor::new(&mut buf));
        for frame in &frames {
            let coords: Vec<[f32; 3]> = atom_indices
                .iter()
                .map(|&i| frame.positions[3 * i..3 * i + 3].try_into().unwrap())
                .collect();
            writer.write_frame_parts(frame.step, frame.time, frame.boxvec, coords.iter(), frame.precision)?;
        }
    }

    let roundtrip = XTCReader::new(Cursor::new(&buf)).read_all_frames()?;

    assert_eq!(frames.len(), roundtrip.len());
    for (i, (orig, rt)) in frames.iter().zip(roundtrip.iter()).enumerate() {
        assert_eq!(orig.step, rt.step, "frame {i}: step mismatch");
        assert_eq!(orig.time, rt.time, "frame {i}: time mismatch");
        assert_eq!(orig.boxvec, rt.boxvec, "frame {i}: boxvec mismatch");
        assert_eq!(rt.positions.len(), atom_indices.len() * 3, "frame {i}: position count mismatch");
        for (j, &atom_idx) in atom_indices.iter().enumerate() {
            for k in 0..3 {
                let orig_val = orig.positions[3 * atom_idx + k];
                let rt_val = rt.positions[3 * j + k];
                let diff = (orig_val - rt_val).abs();
                assert!(
                    diff < 1e-5,
                    "frame {i}, atom {atom_idx}, coord {k}: mismatch ({orig_val} vs {rt_val}, diff {diff})"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn report_compressed_sizes() -> io::Result<()> {
    let inputs = [
        ("ADK", trajectories::ADK),
        ("AUX", trajectories::AUX),
        ("COB", trajectories::COB),
        ("SMOL", trajectories::SMOL),
        ("TEN", trajectories::TEN),
        ("XYZ", trajectories::XYZ),
        ("DELINYAH", trajectories::DELINYAH),
    ];
    for (name, path) in inputs {
        let size = encode_size(path)?;
        println!("{name}: {size} bytes");
    }
    Ok(())
}
