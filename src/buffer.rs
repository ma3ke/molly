use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use crate::reader::read_nbytes;
use crate::{padding, Magic};

pub trait Buffered<'s, 'r, R>: Sized {
    // TODO(buffered): Consider giving the n_bytes from the outside?
    /// Create a new [`Buffer`] reader.
    ///
    /// # Note
    ///
    /// Expects that the first `u32` represents the number of upcoming bytes in the compressed data
    /// stream. If this function is called on a reader that is not at that spot in its stream, the
    /// resulting [`Buffer`] is invalid. This is the same requirement [`read_opaque`] has.
    // We initialize on a Vec<u8> but after preparing this Vec we store the allocation internally
    // as a mutable byte slice, since we do not need to do any Vec-specific operations on it
    // afterwards. When this type is dropped, the ownership of `scratch` is returned since the
    // reference to it dissolves.
    fn new(scratch: &'s mut Vec<u8>, reader: &'r mut R, magic: Magic) -> io::Result<Self>;

    /// Pop a byte from the buffer.
    ///
    /// # Panics
    ///
    /// If the `index` exceeds the number of bytes this [`Buffer`] can read, this function panics.
    /// In that case something is seriously wrong anyway.
    fn pop(&mut self) -> u8;

    /// Returns the byte position of the reader.
    fn tell(&self) -> usize;

    /// Finish will eat your reader, leaving it at the start of the next frame, and then drops it.
    ///
    /// For an implementation that relies on [`std::io::Seek`] ([`Buffer`] in our case), this
    /// really matters.
    fn finish(self) -> io::Result<()>;
}

/// A specialized buffered reader for the compressed datastream.
pub struct Buffer<'s, 'r> {
    /// Internal scratch buffer to read into.
    ///
    /// # Warning
    ///
    /// Accessing bytes from this buffer directly is valid iff the index of that byte < `self.idx`.
    scratch: &'s mut [u8],
    /// Points to the next unfilled byte in `scratch`.
    ///
    /// The starting point for reading bytes from `reader` into `scratch`.
    front: usize,
    /// Points to the last-most byte that has been read.
    head: usize,
    reader: &'r mut File,
    // TODO(buffered): Add some notion of a 'rich' heuristic. For instance, if we know there are
    // 1000 atoms, and we only want to read up until the 500th atom, we can pretty safely assume
    // that we can just read (500/1000) * 1.1 * nbytes = 0.55 * nbytes and be fine.
}

impl Buffer<'_, '_> {
    const BLOCK_SIZE: usize = 0x20000;
    const MIN_BUFFERED_SIZE: usize = 2 * Self::BLOCK_SIZE;

    /// Returns the size of this [`Buffer`].
    const fn size(&self) -> usize {
        self.scratch.len()
    }

    /// Returns the number of bytes that are yet to be read by this [`Buffer`].
    const fn left(&self) -> usize {
        self.size() - self.front
    }

    /// Read enough bytes such that `index` points to a valid byte.
    ///
    /// After this function completes successfully, we can guarantee the following:
    ///
    /// - `index` < `self.front`, because of the inverse condition in the while loop.
    /// - Values before `self.front` are loaded with valid values from the reader.
    #[cold]
    fn read_to_include(&mut self, index: usize) -> io::Result<()> {
        while index >= self.front {
            // TODO(buffered): Consider dealing with n_bytes == 0 indicating eof.
            // Read a bunch of bytes limited by the size of the scratch buffer and BLOCK_SIZE.
            // We would rather do a couple more smaller reads (BLOCK_SIZE) than one big one that
            // goes way beyond what we need according to some AtomSelection.
            let until = usize::min(self.size(), index + Self::BLOCK_SIZE);
            self.front += self.reader.read(&mut self.scratch[self.front..until])?;
        }
        assert!(index < self.front); // Already proven by the while loop, but let's double-check :)
        Ok(())
    }
}

impl<'s, 'r> Buffered<'s, 'r, File> for Buffer<'s, 'r> {
    fn new(scratch: &'s mut Vec<u8>, reader: &'r mut File, magic: Magic) -> io::Result<Self> {
        let count = read_nbytes(reader, magic)?;

        // Fill the scratch buffer with a cautionary value.
        scratch.resize(count + padding(count), 0xff); // FIXME: Is MaybeUninit a good idea here?

        let mut buffer = Self {
            scratch,
            front: 0,
            head: 0,
            reader,
        };

        // In case the buffer size is rather low, it is probably most efficient to just read it all
        // at once, right here.
        if buffer.scratch.len() <= Self::MIN_BUFFERED_SIZE {
            buffer.read_to_include(count.saturating_sub(1))?;
            assert_eq!(buffer.size(), buffer.front)
        }

        Ok(buffer)
    }

    #[inline(always)]
    fn pop(&mut self) -> u8 {
        // If we're out of bytes, we'll have to read new ones.
        // NOTE: This branch is pretty much singularly responsible for the performance difference
        // between unbuffered and buffered decompression (cf. the impl of this function for
        // `UnBuffered`).
        let head = self.head;
        if head >= self.front {
            // FIXME(buffered): For now, let's just fuck this up with a terrible unwrap here.
            // Gotta change this to be io::Result at some point? If we can muster the perf hit
            // at least...
            self.read_to_include(head).unwrap();
        }
        self.head += 1;
        // Safety: We know that `head < self.front`, and that values before `self.front` are valid
        // and will exist. These guarantees are upheld and asserted in `read_to_include`.
        unsafe { *self.scratch.get_unchecked(head) }
    }

    fn tell(&self) -> usize {
        self.head
    }

    fn finish(self) -> io::Result<()> {
        self.reader.seek(SeekFrom::Current(self.left() as i64))?;
        Ok(())
    }
}

pub struct UnBuffered<'s> {
    head: usize,
    scratch: &'s [u8],
}

/// A fallback non-buffered implementation in case [`std::io::Seek`] is not available for `R`.
impl<'s, 'r, R: Read> Buffered<'s, 'r, R> for UnBuffered<'s> {
    fn new(scratch: &'s mut Vec<u8>, reader: &'r mut R, magic: Magic) -> io::Result<Self> {
        let count = read_nbytes(reader, magic)?;
        scratch.resize(count + padding(count), 0);
        reader.read_exact(scratch)?;
        Ok(Self { head: 0, scratch })
    }

    #[inline(always)]
    fn pop(&mut self) -> u8 {
        let head = self.head;
        self.head += 1;
        // Safety: The buffer is pre-allocated to include padding, and the XTC format
        // guarantees we won't read beyond the compressed data size.
        debug_assert!(head < self.scratch.len(), "pop beyond buffer bounds");
        unsafe { *self.scratch.get_unchecked(head) }
    }

    fn tell(&self) -> usize {
        self.head
    }

    fn finish(self) -> io::Result<()> {
        Ok(()) // Nothing to do, since we already read everything.
    }
}
