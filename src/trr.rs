//! TRR file reading and writing.
//!
//! TRR is GROMACS's lossless trajectory format. It stores raw XDR floats (no compression)
//! and supports positions, velocities, and forces per frame.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::reader::{read_f32, read_f64, read_i32};
use crate::BoxVec;
use crate::Position;

const TRR_MAGIC: i32 = 1993;
const TRR_VERSION: &str = "GMX_trn_file";

// ── helpers ─────────────────────────────────────────────────────────────────

fn write_i32(w: &mut impl Write, v: i32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

fn skip_bytes(r: &mut impl Read, n: usize) -> io::Result<()> {
    // Read and discard n bytes without allocating when n is small.
    let mut buf = [0u8; 256];
    let mut remaining = n;
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        r.read_exact(&mut buf[..to_read])?;
        remaining -= to_read;
    }
    Ok(())
}

// ── public types ─────────────────────────────────────────────────────────────

/// Per-frame metadata from a TRR file header.
#[derive(Debug, Clone, PartialEq)]
pub struct TRRHeader {
    pub natoms: usize,
    pub step: u32,
    pub time: f32,
    pub lambda: f32,
    pub has_box: bool,
    pub has_positions: bool,
    pub has_velocities: bool,
    pub has_forces: bool,
    /// True if the file was written with double precision (f64).
    /// Always `false` for files written by [`TRRWriter`].
    pub b_double: bool,
}

/// A single frame read from a TRR file.
#[derive(Debug, Clone, PartialEq)]
pub struct TRRFrame {
    pub step: u32,
    /// Time in picoseconds.
    pub time: f32,
    pub lambda: f32,
    /// Box vectors (column-major, 9 floats).
    pub boxvec: BoxVec,
    /// Flat 3N array of positions (nanometers).
    pub positions: Vec<f32>,
    /// Flat 3N array of velocities (nm/ps), if present.
    pub velocities: Option<Vec<f32>>,
    /// Flat 3N array of forces (kJ mol⁻¹ nm⁻¹), if present.
    pub forces: Option<Vec<f32>>,
}

impl TRRFrame {
    /// Returns the number of atoms in this frame.
    pub fn natoms(&self) -> usize {
        self.positions.len() / 3
    }

    /// Returns an iterator over the coordinates stored in this frame.
    pub fn coords(&self) -> impl Iterator<Item = Position> + '_ {
        self.positions
            .chunks_exact(3)
            .map(|p| p.try_into().unwrap())
    }
}

impl Default for TRRFrame {
    fn default() -> Self {
        TRRFrame {
            step: 0,
            time: 0.0,
            lambda: 0.0,
            boxvec: [0.0; 9],
            positions: Vec::new(),
            velocities: None,
            forces: None,
        }
    }
}

// ── header I/O ───────────────────────────────────────────────────────────────

fn read_trr_header(r: &mut impl Read) -> io::Result<TRRHeader> {
    // magic
    let magic = read_i32(r)?;
    if magic != TRR_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid TRR magic number {magic} (expected {TRR_MAGIC})"
            ),
        ));
    }

    // slen (string length including null — always 13)
    let _slen = read_i32(r)?;
    // str_len (actual string length — always 12)
    let str_len = read_i32(r)? as usize;
    // version string (12 bytes, XDR-aligned to 4-byte boundary → 12 bytes, no padding needed)
    let mut version_bytes = [0u8; 12];
    r.read_exact(&mut version_bytes[..str_len])?;
    // consume XDR padding
    let pad = (4 - (str_len % 4)) % 4;
    skip_bytes(r, pad)?;

    // size fields
    let _ir_size = read_i32(r)?;
    let _e_size = read_i32(r)?;
    let box_size = read_i32(r)? as usize;
    let _vir_size = read_i32(r)?;
    let _pres_size = read_i32(r)?;
    let _top_size = read_i32(r)?;
    let _sym_size = read_i32(r)?;
    let x_size = read_i32(r)? as usize;
    let v_size = read_i32(r)? as usize;
    let f_size = read_i32(r)? as usize;

    // natoms, step, nre
    let natoms = read_i32(r)? as usize;
    let step = read_i32(r)? as u32;
    let _nre = read_i32(r)?;

    // detect precision: if box is 72 bytes or x is natoms*3*8, it's double
    let b_double = box_size == 72
        || (natoms > 0 && x_size == natoms * 3 * 8)
        || (natoms > 0 && v_size == natoms * 3 * 8)
        || (natoms > 0 && f_size == natoms * 3 * 8);

    // time and lambda
    let (time, lambda) = if b_double {
        let t = read_f64(r)? as f32;
        let l = read_f64(r)? as f32;
        (t, l)
    } else {
        let t = read_f32(r)?;
        let l = read_f32(r)?;
        (t, l)
    };

    Ok(TRRHeader {
        natoms,
        step,
        time,
        lambda,
        has_box: box_size > 0,
        has_positions: x_size > 0,
        has_velocities: v_size > 0,
        has_forces: f_size > 0,
        b_double,
    })
}

struct TRRWriteParams {
    natoms: usize,
    step: u32,
    time: f32,
    lambda: f32,
    has_box: bool,
    has_positions: bool,
    has_velocities: bool,
    has_forces: bool,
}

fn write_trr_header(w: &mut impl Write, p: &TRRWriteParams) -> io::Result<()> {
    // Always write single precision (f32).
    let float_size = 4usize;
    let vec3_bytes = p.natoms * 3 * float_size;

    write_i32(w, TRR_MAGIC)?;
    // slen: length of version string + 1 (for null terminator)
    write_i32(w, TRR_VERSION.len() as i32 + 1)?;
    // str_len: actual string length
    write_i32(w, TRR_VERSION.len() as i32)?;
    // version string (12 bytes, already 4-byte aligned)
    w.write_all(TRR_VERSION.as_bytes())?;
    // XDR padding for string
    let pad = (4 - (TRR_VERSION.len() % 4)) % 4;
    w.write_all(&[0u8; 4][..pad])?;

    write_i32(w, 0)?; // ir_size
    write_i32(w, 0)?; // e_size
    write_i32(w, if p.has_box { 9 * float_size as i32 } else { 0 })?; // box_size
    write_i32(w, 0)?; // vir_size
    write_i32(w, 0)?; // pres_size
    write_i32(w, 0)?; // top_size
    write_i32(w, 0)?; // sym_size
    write_i32(w, if p.has_positions { vec3_bytes as i32 } else { 0 })?; // x_size
    write_i32(w, if p.has_velocities { vec3_bytes as i32 } else { 0 })?; // v_size
    write_i32(w, if p.has_forces { vec3_bytes as i32 } else { 0 })?; // f_size

    write_i32(w, p.natoms as i32)?;
    write_i32(w, p.step as i32)?;
    write_i32(w, 0)?; // nre

    write_f32(w, p.time)?;
    write_f32(w, p.lambda)?;

    Ok(())
}

// ── data I/O helpers ─────────────────────────────────────────────────────────

/// Read `n` Vec3 values (each as f32 or f64) into a flat f32 Vec.
fn read_vec3s(r: &mut impl Read, n: usize, double: bool) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n * 3];
    if double {
        for v in &mut out {
            *v = read_f64(r)? as f32;
        }
    } else {
        for v in &mut out {
            *v = read_f32(r)?;
        }
    }
    Ok(out)
}

/// Write a flat f32 slice as XDR big-endian f32 values.
fn write_vec3s(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    for &v in data {
        write_f32(w, v)?;
    }
    Ok(())
}

/// Write Vec3s from an iterator.
fn write_vec3s_iter<'a>(
    w: &mut impl Write,
    iter: impl ExactSizeIterator<Item = &'a [f32; 3]>,
) -> io::Result<()> {
    for pos in iter {
        write_f32(w, pos[0])?;
        write_f32(w, pos[1])?;
        write_f32(w, pos[2])?;
    }
    Ok(())
}

// ── TRRReader ────────────────────────────────────────────────────────────────

/// Reader for TRR trajectory files.
pub struct TRRReader<R> {
    pub file: R,
    pub step: usize,
}

impl TRRReader<File> {
    /// Open a TRR file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self::new(file))
    }

    /// Reset to the first frame.
    pub fn home(&mut self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.step = 0;
        Ok(())
    }

    /// Skip frames until a frame where `time >= t` is found.
    ///
    /// Returns the header of the first matching frame and positions the reader at the start of
    /// that frame.
    pub fn skip_to_time(&mut self, t: f32) -> io::Result<TRRHeader> {
        loop {
            let pos = self.file.stream_position()?;
            let header = read_trr_header(&mut self.file)?;
            if header.time >= t {
                self.file.seek(SeekFrom::Start(pos))?;
                return Ok(header);
            }
            skip_frame_data(&mut self.file, &header)?;
        }
    }

    /// Skip forward `n` frames.
    pub fn skip_frames(&mut self, n: u64) -> io::Result<()> {
        for _ in 0..n {
            let header = read_trr_header(&mut self.file)?;
            skip_frame_data(&mut self.file, &header)?;
        }
        Ok(())
    }

    /// Read the last frame in the trajectory.
    pub fn read_last_frame(&mut self, frame: &mut TRRFrame) -> io::Result<()> {
        // Scan from the beginning to find the last frame.
        self.home()?;
        let mut last = TRRFrame::default();
        let mut found = false;
        loop {
            match self.read_frame(&mut last) {
                Ok(()) => found = true,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
        if found {
            *frame = last;
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no frames found"))
        }
    }
}

impl<R: Read> TRRReader<R> {
    /// Create a new TRR reader wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            file: reader,
            step: 0,
        }
    }

    /// Read the next frame from the trajectory.
    pub fn read_frame(&mut self, frame: &mut TRRFrame) -> io::Result<()> {
        let header = read_trr_header(&mut self.file)?;

        // box
        frame.boxvec = [0.0; 9];
        if header.has_box {
            let data = read_vec3s(&mut self.file, 3, header.b_double)?;
            frame.boxvec.copy_from_slice(&data);
        }

        // positions
        if header.has_positions {
            frame.positions = read_vec3s(&mut self.file, header.natoms, header.b_double)?;
        } else {
            frame.positions.clear();
        }

        // velocities
        if header.has_velocities {
            frame.velocities = Some(read_vec3s(&mut self.file, header.natoms, header.b_double)?);
        } else {
            frame.velocities = None;
        }

        // forces
        if header.has_forces {
            frame.forces = Some(read_vec3s(&mut self.file, header.natoms, header.b_double)?);
        } else {
            frame.forces = None;
        }

        frame.step = header.step;
        frame.time = header.time;
        frame.lambda = header.lambda;
        self.step += 1;

        Ok(())
    }

    /// Read all frames from the trajectory.
    pub fn read_all_frames(&mut self) -> io::Result<Box<[TRRFrame]>> {
        let mut frames = Vec::new();
        loop {
            let mut frame = TRRFrame::default();
            match self.read_frame(&mut frame) {
                Ok(()) => frames.push(frame),
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
        Ok(frames.into_boxed_slice())
    }
}

/// Skip over the data section of a frame (positions, velocities, forces, box).
fn skip_frame_data(r: &mut impl Read, header: &TRRHeader) -> io::Result<()> {
    let float_bytes = if header.b_double { 8 } else { 4 };
    let vec9 = 9 * float_bytes;
    let vecn = header.natoms * 3 * float_bytes;

    if header.has_box {
        skip_bytes(r, vec9)?;
    }
    if header.has_positions {
        skip_bytes(r, vecn)?;
    }
    if header.has_velocities {
        skip_bytes(r, vecn)?;
    }
    if header.has_forces {
        skip_bytes(r, vecn)?;
    }
    Ok(())
}

// ── TRRWriter ────────────────────────────────────────────────────────────────

/// Writer for TRR trajectory files.
pub struct TRRWriter<W> {
    pub file: W,
}

impl TRRWriter<File> {
    /// Create a new TRR file at the given path.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self::new(file))
    }
}

impl<W: Write> TRRWriter<W> {
    /// Create a new TRR writer wrapping the given writer.
    pub fn new(writer: W) -> Self {
        Self { file: writer }
    }

    /// Write a frame to the TRR file.
    pub fn write_frame(&mut self, frame: &TRRFrame) -> io::Result<()> {
        let natoms = frame.natoms();
        let has_velocities = frame.velocities.is_some();
        let has_forces = frame.forces.is_some();

        write_trr_header(&mut self.file, &TRRWriteParams {
            natoms,
            step: frame.step,
            time: frame.time,
            lambda: frame.lambda,
            has_box: true,
            has_positions: !frame.positions.is_empty(),
            has_velocities,
            has_forces,
        })?;

        // box
        write_vec3s(&mut self.file, &frame.boxvec)?;

        // positions
        if !frame.positions.is_empty() {
            write_vec3s(&mut self.file, &frame.positions)?;
        }

        // velocities
        if let Some(ref v) = frame.velocities {
            write_vec3s(&mut self.file, v)?;
        }

        // forces
        if let Some(ref f) = frame.forces {
            write_vec3s(&mut self.file, f)?;
        }

        Ok(())
    }

    /// Write frame data from iterators.
    ///
    /// Coordinates, velocities, and forces are sourced from [`ExactSizeIterator`]s yielding
    /// `&[f32; 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_frame_parts<'a>(
        &mut self,
        step: u32,
        time: f32,
        lambda: f32,
        boxvec: BoxVec,
        positions: impl ExactSizeIterator<Item = &'a [f32; 3]>,
        velocities: Option<impl ExactSizeIterator<Item = &'a [f32; 3]>>,
        forces: Option<impl ExactSizeIterator<Item = &'a [f32; 3]>>,
    ) -> io::Result<()> {
        let natoms = positions.len();
        let has_velocities = velocities.is_some();
        let has_forces = forces.is_some();

        write_trr_header(&mut self.file, &TRRWriteParams {
            natoms,
            step,
            time,
            lambda,
            has_box: true,
            has_positions: natoms > 0,
            has_velocities,
            has_forces,
        })?;

        // box
        write_vec3s(&mut self.file, &boxvec)?;

        // positions
        write_vec3s_iter(&mut self.file, positions)?;

        // velocities
        if let Some(vels) = velocities {
            write_vec3s_iter(&mut self.file, vels)?;
        }

        // forces
        if let Some(frc) = forces {
            write_vec3s_iter(&mut self.file, frc)?;
        }

        Ok(())
    }
}
