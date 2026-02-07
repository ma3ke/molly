use bencher::{benchmark_group, benchmark_main, Bencher};
use molly::{Frame, XTCReader, XTCWriter};

benchmark_main!(encoding);
benchmark_group!(
    encoding,
    write_compressed_positions,
    write_compressed_positions_direct,
    write_compressed_positions_xdrfile
);

const PATH: &str = "tests/trajectories/adk_oplsaa.xtc";

fn write_compressed_positions(b: &mut Bencher) {
    let mut reader = XTCReader::open(PATH).unwrap();
    let mut frame = Frame::default();
    reader.read_frame(&mut frame).unwrap();

    let tmp_path = std::env::temp_dir().join("bench_molly.xtc");

    b.iter(|| {
        let mut writer = XTCWriter::create(&tmp_path).unwrap();
        writer.write_frame(&frame).unwrap();
    });

    let _ = std::fs::remove_file(&tmp_path);
}

fn write_compressed_positions_direct(b: &mut Bencher) {
    use std::io::BufWriter;

    let mut reader = XTCReader::open(PATH).unwrap();
    let mut frame = Frame::default();
    reader.read_frame(&mut frame).unwrap();

    let tmp_path = std::env::temp_dir().join("bench_molly_direct.xtc");

    b.iter(|| {
        let file = std::fs::File::create(&tmp_path).unwrap();
        let writer = BufWriter::new(file);
        let mut xtc_writer = XTCWriter::new(writer);
        xtc_writer.write_frame(&frame).unwrap();
    });

    let _ = std::fs::remove_file(&tmp_path);
}

fn write_compressed_positions_xdrfile(b: &mut Bencher) {
    use xdrfile::Trajectory;

    let mut xdr_reader = xdrfile::XTCTrajectory::open_read(PATH).unwrap();
    let natoms = xdr_reader.get_num_atoms().unwrap();
    let mut xdr_frame = xdrfile::Frame::with_len(natoms);
    xdr_reader.read(&mut xdr_frame).unwrap();

    let tmp_path = std::env::temp_dir().join("bench_xdrfile.xtc");

    b.iter(|| {
        let mut trajectory = xdrfile::XTCTrajectory::open_write(&tmp_path).unwrap();
        trajectory.write(&xdr_frame).unwrap();
    });

    let _ = std::fs::remove_file(&tmp_path);
}
