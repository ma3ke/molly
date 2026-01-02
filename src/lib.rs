use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::{cell::Cell, path::Path};

use glam::{Mat3, Vec3};
use reader::read_nbytes;

use crate::buffer::{Buffer, UnBuffered};
use crate::reader::{
    read_boxvec, read_compressed_positions, read_f32, read_f32s, read_i32, read_u32,
};
use crate::selection::{AtomSelection, FrameSelection};

pub mod buffer;
pub mod reader;
pub mod selection;

// See https://gitlab.com/gromacs/gromacs/-/blob/v2024.1/src/gromacs/fileio/xdrf.h?ref_type=tags#L78
pub const XTC_1995_MAX_NATOMS: usize = 298261617;

thread_local! {
    /// A scratch buffer to read encoded bytes into for subsequent decoding.
    static SCRATCH: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

pub type BoxVec = Mat3;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Magic {
    Xtc1995 = 1995,
    Xtc2023 = 2023,
}

impl Magic {
    pub const XTC_1995: i32 = Magic::Xtc1995 as _;
    pub const XTC_2023: i32 = Magic::Xtc2023 as _;

    fn to_be_bytes(&self) -> [u8; 4] {
        (*self as i32).to_be_bytes()
    }
}

impl TryFrom<i32> for Magic {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            Magic::XTC_1995 => Ok(Self::Xtc1995),
            Magic::XTC_2023 => Ok(Self::Xtc2023),
            unknown => Err(format!(
                "found invalid magic number '{unknown}' ({unknown:#0x}), {} and {} are supported",
                Self::XTC_1995,
                Self::XTC_2023
            )),
        }
    }
}

impl std::fmt::Display for Magic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", *self as i32)
    }
}

/// The header of a single xtc frame.
pub struct Header {
    pub magic: Magic,
    pub natoms: usize,
    pub step: u32,
    pub time: f32,
    pub boxvec: Mat3,
    pub natoms_repeated: usize,
}

impl Header {
    pub const SIZE: usize = 4 * (5 + 9);

    pub fn read(file: &mut impl Read) -> io::Result<Self> {
        let magic = Magic::try_from(read_i32(file)?)
            .map_err(|err| io::Error::other(format!("could not read header: {err}")))?;
        let natoms: usize = read_u32(file)?
            .try_into()
            .map_err(|err| io::Error::other(format!("could not read natoms: {err}")))?;
        let step: u32 = read_u32(file)?;
        let time = read_f32(file)?;

        // Read the frame data.
        let boxvec = read_boxvec(file)?;
        let natoms_repeated = read_u32(file)?
            .try_into()
            .map_err(|err| io::Error::other(format!("could not read second natoms: {err}")))?;
        assert_eq!(natoms, natoms_repeated);

        Ok(Header {
            magic,
            natoms,
            step,
            time,
            boxvec,
            natoms_repeated,
        })
    }

    pub fn to_be_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = Vec::new();
        bytes.extend(self.magic.to_be_bytes()); // i32

        let natoms = u32::try_from(self.natoms).unwrap().to_be_bytes();
        bytes.extend(natoms); // u32
        bytes.extend(i32::try_from(self.step).unwrap().to_be_bytes()); // i32
        bytes.extend(self.time.to_be_bytes()); // f32
        bytes.extend(
            self.boxvec
                .to_cols_array()
                .map(f32::to_be_bytes)
                .iter()
                .flatten(),
        ); // 9 × f32
        assert_eq!(self.natoms, self.natoms_repeated);
        bytes.extend(natoms); // u32

        bytes.try_into().unwrap()
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Frame {
    pub step: u32,
    /// Time in picoseconds.
    pub time: f32,
    pub boxvec: BoxVec,
    pub precision: f32,
    pub positions: Vec<f32>,
}

impl Frame {
    /// Returns an iterator over the coordinates stored in this [`Frame`].
    pub fn coords(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.positions.chunks_exact(3).map(Vec3::from_slice)
    }

    /// Returns the number of atoms in this [`Frame`].
    pub fn natoms(&self) -> usize {
        let npos = self.positions.len();
        assert_eq!(
            npos % 3,
            0,
            "the number of single positions in a frame must always be a multiple of 3"
        );
        npos / 3
    }
}

/// Calculate the xdr padding for some number of bytes.
#[doc(hidden)]
pub fn padding(n: usize) -> usize {
    (4 - (n % 4)) % 4
}

/// Read the positions in a frame after the header.
///
/// If successful, returns the number of compressed bytes that were read.
///
/// Internal use.
#[doc(hidden)]
pub fn read_positions<'s, 'r, B: buffer::Buffered<'s, 'r, R>, R: Read>(
    file: &'r mut R,
    header_natoms: usize,
    scratch: &'s mut Vec<u8>,
    frame: &mut Frame,
    atom_selection: &AtomSelection,
    magic: Magic,
) -> io::Result<usize> {
    // If the atom_selection specifies fewer atoms, we will only allocate up to that point.
    let natoms_selected = atom_selection.natoms_selected(header_natoms);

    // Resize the positions array for the selected number of atoms.
    frame.positions.resize(natoms_selected * 3, f32::NAN);
    frame.precision = read_f32(file)?;
    read_compressed_positions::<B, R>(
        file,
        header_natoms,
        &mut frame.positions,
        frame.precision,
        scratch,
        atom_selection,
        magic,
    )
}

#[derive(Debug, Clone)]
pub struct XTCReader<R> {
    pub file: R,
    pub step: usize,
}

impl XTCReader<std::fs::File> {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self::new(file))
    }
}

impl<R: Read> XTCReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            file: reader,
            step: 0,
        }
    }

    /// Read the header at the start of a frame.
    ///
    /// Assumes the internal reader is at the start of a new frame header.
    pub fn read_header(&mut self) -> io::Result<Header> {
        Header::read(&mut self.file)
    }

    /// Read a small number of uncompressed positions.
    ///
    /// If successful, returns the number of compressed bytes that were read.
    ///
    /// # Panics
    ///
    /// `natoms` must be 9 or less, otherwise the positions must be decompressed and cannot be read
    /// directly through this function.  
    ///
    /// Oh xtc, you are so fucking weird.
    pub fn read_smol_positions(
        &mut self,
        natoms: usize,
        frame: &mut Frame,
        atom_selection: &AtomSelection,
    ) -> io::Result<usize> {
        assert!(
            natoms <= 9,
            "only read uncomprossed positions when the number of atoms is 9 or less"
        );

        // In case the number of atoms is very small, just read their uncompressed positions.
        frame.positions.resize(natoms * 3, 0.0);
        let mut buf = [0.0; 9 * 3]; // We have at most 9 atoms, so we handle them on the stack.
        let buf = &mut buf[..natoms * 3];
        read_f32s(&mut self.file, buf)?;
        frame.positions.clear();
        frame.positions.extend(
            buf.chunks_exact(3)
                .enumerate()
                .filter_map(|(idx, pos): (usize, &[f32])| -> Option<[f32; 3]> {
                    if atom_selection.is_included(idx).unwrap_or_default() {
                        Some(pos.try_into().unwrap())
                    } else {
                        None
                    }
                })
                .flatten(),
        );
        // TODO: It is unclear to me what to do with the precision in this case. It is
        // basically invalid, or just irrelevant here, since we don't decode them. They were
        // never compressed to begin with.

        Ok(buf.len() * std::mem::size_of::<f32>())
    }

    /// A convenience function to read all frames in a trajectory.
    ///
    /// It is likely more efficient to use [`XTCReader::read_frame`] if you are only interested in
    /// the values of a single frame at a time.
    pub fn read_all_frames(&mut self) -> io::Result<Box<[Frame]>> {
        let mut frames = Vec::new();
        loop {
            let mut frame = Frame::default();
            if let Err(err) = self.read_frame(&mut frame) {
                match err.kind() {
                    // We have found the end of the file. No more frames, we're done.
                    io::ErrorKind::UnexpectedEof => break,
                    // Something else went wrong...
                    _ => Err(err)?,
                }
            }
            frames.push(frame);
        }
        Ok(frames.into_boxed_slice())
    }

    /// Reads and returns a [`Frame`] and advances one step.
    pub fn read_frame(&mut self, frame: &mut Frame) -> io::Result<()> {
        self.read_frame_with_selection(frame, &AtomSelection::All)
    }

    /// Reads and returns a [`Frame`] according to the [`AtomSelection`], and advances one step.
    pub fn read_frame_with_selection(
        &mut self,
        frame: &mut Frame,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        // Take the thread-local SCRATCH and use that while decoding the values.
        let mut scratch = SCRATCH.take();
        self.read_frame_with_scratch(frame, &mut scratch, atom_selection)
    }

    /// Reads and returns a [`Frame`] and advances one step, internally reading the compressed data
    /// into `scratch`.
    ///
    /// # Note
    ///
    /// This function performs the work of [`XTCReader::read_frame`], but leaves all allocations to
    /// the caller.
    ///
    /// The contents of `scratch` should not be depended upon! It just serves as a scratch buffer
    /// for the inner workings of decoding.
    ///
    /// In most cases, [`XTCReader::read_frame`] is more than sufficient. This function only serves
    /// to make specific optimization possible.
    pub fn read_frame_with_scratch(
        &mut self,
        frame: &mut Frame,
        scratch: &mut Vec<u8>,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        self.read_frame_with_scratch_impl::<UnBuffered>(frame, scratch, atom_selection)
    }

    /// Implementation of reading a frame with a scratch buffer.
    fn read_frame_with_scratch_impl<'s, 'r, B: buffer::Buffered<'s, 'r, R>>(
        &'r mut self,
        frame: &mut Frame,
        scratch: &'s mut Vec<u8>,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        // Start of by reading the header.
        let header = self.read_header()?;

        // Now, we read the atoms.
        if header.natoms <= 9 {
            self.read_smol_positions(header.natoms, frame, atom_selection)?;
        } else {
            read_positions::<B, R>(
                &mut self.file,
                header.natoms,
                scratch,
                frame,
                atom_selection,
                header.magic,
            )?;
        }

        self.step += 1;

        frame.step = header.step;
        frame.time = header.time;
        frame.boxvec = header.boxvec;

        Ok(())
    }
}

impl XTCReader<File> {
    /// Reset the reader to its initial position.
    ///
    /// Go back to the first frame.
    pub fn home(&mut self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.step = 0;
        Ok(())
    }

    /// Returns the offsets from the headers in this [`XTCReader<R>`] from its current position.
    ///
    /// The last value points one byte after the last byte in the reader.
    ///
    /// If this function is called when the internal reader is not at its starting position, the
    /// frame offsets _from_ its position are determined. If you wish to determine the offsets from
    /// the initial reader position, call [`XTCReader::home`] before calling this function.
    ///
    /// # Errors
    ///
    /// This function will pass through any reader errors.
    pub fn determine_offsets_exclusive(&mut self, until: Option<usize>) -> io::Result<Box<[u64]>> {
        let file = &mut self.file;
        // Remember where we start so we can return to it later.
        let start_pos = file.stream_position()?;

        let mut offsets = Vec::new();

        while until.map_or(true, |until| offsets.len() < until) {
            let header = match Header::read(file) {
                Ok(header) => header,
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => Err(err)?,
            };

            let skip = if header.natoms <= 9 {
                // Know how many bytes are in this frame until the next header since the positions
                // are uncompressed.
                header.natoms as u64 * 3 * 4
            } else {
                // We need to read the nbytes value to get the offset until the next header.
                file.seek(SeekFrom::Current(32))?;
                // The size of the buffer is stored either as a 64 or 32-bit integer, depending on
                // the magic number in the header.
                let nbytes = read_nbytes(file, header.magic)? as u64;
                nbytes + padding(nbytes as usize) as u64
            };
            let offset = file.seek(SeekFrom::Current(skip as i64))?;
            offsets.push(offset);
        }

        // Return back to where we started.
        file.seek(SeekFrom::Start(start_pos))?;

        Ok(offsets.into_boxed_slice())
    }

    /// Returns the offsets of this [`XTCReader<R>`] from its current position.
    ///
    /// The last value points to the start of the last frame.
    ///
    /// If this function is called when the internal reader is not at its starting position, the
    /// frame offsets _from_ its position are determined. If you wish to determine the offsets from
    /// the initial reader position, call [`XTCReader::home`] before calling this function.
    ///
    /// # Errors
    ///
    /// This function will pass through any reader errors.
    pub fn determine_offsets(&mut self, until: Option<usize>) -> io::Result<Box<[u64]>> {
        let mut offsets = vec![0];
        let exclusive = self.determine_offsets_exclusive(until)?;
        offsets.extend(exclusive.iter().take(exclusive.len().saturating_sub(1)));
        Ok(offsets.into_boxed_slice())
    }

    /// Returns the frame sizes of this [`XTCReader<R>`].
    ///
    /// # Errors
    ///
    /// This function will pass through any reader errors.
    pub fn determine_frame_sizes(&mut self, until: Option<usize>) -> io::Result<Box<[u64]>> {
        let starts = self.determine_offsets_exclusive(until)?;
        let ends = starts.iter().clone().skip(1);
        Ok(starts
            .iter()
            .zip(ends)
            .map(|(s, e)| e - s)
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }

    /// Seeks to offset, then reads and returns a [`Frame`] and advances one step.
    ///
    /// # Note
    ///
    /// The `BUFFERED` const generic value can be used to set whether the frame reader will read in
    /// a buffered manner or not at compile time.
    ///
    /// Buffered reading is most favorable when a small number of positions are read from the top of
    /// the frame (leaving many positions that do not need to be read at the bottom), especially at the
    /// point where disk read speed is a bottleneck.
    pub fn read_frame_at_offset<const BUFFERED: bool>(
        &mut self,
        frame: &mut Frame,
        offset: u64,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        match BUFFERED {
            false => self.read_frame_with_selection(frame, atom_selection),
            true => self.read_frame_with_selection_buffered(frame, atom_selection),
        }
    }

    /// Append [`Frame`]s to the `frames` buffer according to a [`Selection`].
    ///
    /// If successful, it will return the number of frames that were read.
    /// This can be useful since the selection itself is not enough to tell how many frames will
    /// actually be read.
    ///
    /// # Note
    ///
    /// The `BUFFERED` const generic value can be used to set whether the frame reader will read in
    /// a buffered manner or not at compile time.
    ///
    /// Buffered reading is most favorable when a small number of positions are read from the top of
    /// the frame (leaving many positions that do not need to be read at the bottom), especially at the
    /// point where disk read speed is a bottleneck.
    pub fn read_frames<const BUFFERED: bool>(
        &mut self,
        frames: &mut impl Extend<Frame>,
        frame_selection: &FrameSelection,
        atom_selection: &AtomSelection,
    ) -> io::Result<usize> {
        let offsets = self.determine_offsets(frame_selection.until())?;
        let mut n = 0;
        for (idx, &offset) in offsets.iter().enumerate() {
            match frame_selection.is_included(idx) {
                Some(true) => {}
                Some(false) => continue,
                None => break,
            }
            let mut frame = Frame::default();
            self.read_frame_at_offset::<BUFFERED>(&mut frame, offset, atom_selection)?;
            frames.extend(Some(frame));
            n += 1;
        }

        Ok(n)
    }

    /// Reads and returns a [`Frame`] according to the [`AtomSelection`], and advances one step.
    pub fn read_frame_with_selection_buffered(
        &mut self,
        frame: &mut Frame,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        // Take the thread-local SCRATCH and use that while decoding the values.
        let mut scratch = SCRATCH.take();
        self.read_frame_with_scratch_buffered(frame, &mut scratch, atom_selection)
    }

    /// Reads and returns a [`Frame`] and advances one step, internally reading the compressed data
    /// into `scratch`.
    ///
    /// # Note
    ///
    /// This function performs the work of [`XTCReader::read_frame`], but leaves all allocations to
    /// the caller.
    ///
    /// The contents of `scratch` should not be depended upon! It just serves as a scratch buffer
    /// for the inner workings of decoding.
    ///
    /// In most cases, [`XTCReader::read_frame`] is more than sufficient. This function only serves
    /// to make specific optimization possible.
    pub fn read_frame_with_scratch_buffered(
        &mut self,
        frame: &mut Frame,
        scratch: &mut Vec<u8>,
        atom_selection: &AtomSelection,
    ) -> io::Result<()> {
        self.read_frame_with_scratch_impl::<Buffer>(frame, scratch, atom_selection)
    }
}
