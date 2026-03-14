mod common;

use std::io::{self, Cursor};

use common::trajectories;
use molly::{TRRFrame, TRRReader, TRRWriter};

fn assert_trr_frames_eq(original: &[TRRFrame], roundtrip: &[TRRFrame]) {
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
        // TRR is lossless — expect exact equality.
        assert_eq!(orig.positions, rt.positions, "frame {i}: position mismatch");
    }
}

macro_rules! roundtrip_test {
    ($name:ident, $path:expr) => {
        #[test]
        fn $name() -> io::Result<()> {
            let frames = TRRReader::open($path)?.read_all_frames()?;
            let mut buf = Vec::new();
            {
                let mut writer = TRRWriter::new(Cursor::new(&mut buf));
                for frame in frames.iter() {
                    writer.write_frame(frame)?;
                }
            }
            let roundtrip = TRRReader::new(Cursor::new(&buf)).read_all_frames()?;
            assert_trr_frames_eq(&frames, &roundtrip);
            Ok(())
        }
    };
}

roundtrip_test!(roundtrip_adk_trr, trajectories::ADK_TRR);
roundtrip_test!(roundtrip_smol_trr, trajectories::SMOL_TRR);
roundtrip_test!(roundtrip_ten_trr, trajectories::TEN_TRR);

#[test]
fn write_trr_frame_parts_ten() -> io::Result<()> {
    let frames = TRRReader::open(trajectories::TEN_TRR)?.read_all_frames()?;
    let atom_indices: &[usize] = &[1, 3, 7];

    let mut buf = Vec::new();
    {
        let mut writer = TRRWriter::new(Cursor::new(&mut buf));
        for frame in &frames {
            let coords: Vec<[f32; 3]> = atom_indices
                .iter()
                .map(|&i| frame.positions[3 * i..3 * i + 3].try_into().unwrap())
                .collect();
            writer.write_frame_parts(
                frame.step,
                frame.time,
                frame.lambda,
                frame.boxvec,
                coords.iter(),
                None::<std::iter::Empty<&[f32; 3]>>,
                None::<std::iter::Empty<&[f32; 3]>>,
            )?;
        }
    }

    let roundtrip = TRRReader::new(Cursor::new(&buf)).read_all_frames()?;

    assert_eq!(frames.len(), roundtrip.len());
    for (i, (orig, rt)) in frames.iter().zip(roundtrip.iter()).enumerate() {
        assert_eq!(orig.step, rt.step, "frame {i}: step mismatch");
        assert_eq!(orig.time, rt.time, "frame {i}: time mismatch");
        assert_eq!(orig.boxvec, rt.boxvec, "frame {i}: boxvec mismatch");
        assert_eq!(
            rt.positions.len(),
            atom_indices.len() * 3,
            "frame {i}: position count mismatch"
        );
        for (j, &atom_idx) in atom_indices.iter().enumerate() {
            for k in 0..3 {
                let orig_val = orig.positions[3 * atom_idx + k];
                let rt_val = rt.positions[3 * j + k];
                assert_eq!(
                    orig_val, rt_val,
                    "frame {i}, atom {atom_idx}, coord {k}: mismatch ({orig_val} vs {rt_val})"
                );
            }
        }
    }
    Ok(())
}
