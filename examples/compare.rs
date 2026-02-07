use molly::XTCReader;
use xdrfile::{Trajectory, XTCTrajectory};

fn round_to(v: f32, decimals: u32) -> f32 {
    let d = f32::powi(10.0, decimals as i32);
    (v * d).round() / d
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("please provide one xtc trajectory path");
    let decimals: u32 = args.next().and_then(|d| d.parse().ok()).unwrap_or(3);
    dbg!(decimals);

    let mut reader = XTCReader::open(&path)?;
    let mut trajectory = XTCTrajectory::open_read(&path).map_err(std::io::Error::other)?;
    let mut xdr_frame = {
        let natoms = trajectory.get_num_atoms().map_err(std::io::Error::other)?;
        xdrfile::Frame::with_len(natoms)
    };
    let mut n = 0;
    let mut natoms = 0;
    let mut frame = molly::Frame::default();
    while reader.read_frame(&mut frame).is_ok() {
        trajectory
            .read(&mut xdr_frame)
            .map_err(std::io::Error::other)?;

        for (molly_coord, xdrf_coord) in frame.coords().zip(&xdr_frame.coords) {
            let molly_coord = molly_coord.map(|v| round_to(v, decimals));
            let xdrf_coord = xdrf_coord.map(|v| round_to(v, decimals));
            assert_eq!(molly_coord, xdrf_coord);
        }

        natoms = frame.positions.len() / 3;
        n += 1;
    }
    eprintln!("compare: read {n} frames");
    assert_eq!(natoms, xdr_frame.coords.len());
    eprintln!("{} atoms", xdr_frame.coords.len());

    Ok(())
}
