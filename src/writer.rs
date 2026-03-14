use std::io::{self, Write};

use crate::reader::{FIRSTIDX, MAGICINTS};
use crate::{padding, Magic};

/// XDR padding bytes.
const ZERO_PAD: [u8; 3] = [0; 3];

/// Maximum size that can be safely multiplied without overflow in sizeofints.
const MAX_MULTIPLIABLE_SIZE: u32 = 0x00ff_ffff;

/// Maximum run length: 8 coordinate triplets.
const MAX_RUN_COORDS: usize = 8 * 3;

/// Tracks whether the encoding precision should change.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SizeChange {
    Decrease,
    Same,
    Increase,
}

/// Check if all coordinate components are within threshold of each other.
#[inline]
const fn coords_within_threshold(a: [i32; 3], b: [i32; 3], threshold: i32) -> bool {
    (a[0] - b[0]).abs() < threshold
        && (a[1] - b[1]).abs() < threshold
        && (a[2] - b[2]).abs() < threshold
}

#[derive(Default)]
struct EncodeState {
    /// Number of pending bits in lastbyte (0-7).
    lastbits: usize,
    /// Pending bits waiting to be written (stored in low positions).
    lastbyte: u8,
}

/// Encode `nbits` bits from `value` into the buffer.
/// Bits are written MSB-first to match the decoder's expectations.
#[inline]
fn encodebits(buf: &mut Vec<u8>, state: &mut EncodeState, value: u32, nbits: usize) {
    if nbits == 0 {
        return;
    }

    // Combine pending bits with new bits.
    // state.lastbyte has state.lastbits pending bits in low positions.
    let total_bits = state.lastbits + nbits;
    let pending = ((state.lastbyte as u64) << nbits) | (value as u64);

    // Write complete bytes (MSB first).
    let mut remaining_bits = total_bits;
    while remaining_bits >= 8 {
        let shift = remaining_bits - 8;
        let byte = (pending >> shift) as u8;
        buf.push(byte);
        remaining_bits -= 8;
    }

    // Store leftover bits for next call.
    state.lastbits = remaining_bits;
    state.lastbyte = (pending & ((1u64 << remaining_bits) - 1)) as u8;
}

/// Encode a full byte, combining with pending bits.
#[inline]
fn encodebyte(buf: &mut Vec<u8>, state: &mut EncodeState, byte: u8) {
    encodebits(buf, state, byte as u32, 8);
}

/// Flush any remaining pending bits as a final byte.
fn flush_bits(buf: &mut Vec<u8>, state: &mut EncodeState) {
    if state.lastbits > 0 {
        // Pad the pending bits to fill a byte (MSB-aligned).
        buf.push(state.lastbyte << (8 - state.lastbits));
        state.lastbits = 0;
        state.lastbyte = 0;
    }
}

/// Pack three integers into a u32 using the size encoding.
/// Uses wrapping arithmetic since intermediate products may overflow.
#[inline]
const fn pack_into_u32(nums: [i32; 3], sizes: [u32; 3]) -> u32 {
    let sz = sizes[2];
    let szy = sizes[1].wrapping_mul(sz);
    (nums[0] as u32)
        .wrapping_mul(szy)
        .wrapping_add((nums[1] as u32).wrapping_mul(sz))
        .wrapping_add(nums[2] as u32)
}

/// Pack three integers into a u64 using the size encoding.
/// Uses wrapping arithmetic for consistency with u32 version.
#[inline]
const fn pack_into_u64(nums: [i32; 3], sizes: [u32; 3]) -> u64 {
    let sz = sizes[2] as u64;
    let szy = (sizes[1] as u64).wrapping_mul(sz);
    (nums[0] as u64)
        .wrapping_mul(szy)
        .wrapping_add((nums[1] as u64).wrapping_mul(sz))
        .wrapping_add(nums[2] as u64)
}

/// Write a packed value LSB-first with the given number of bits.
fn write_packed_bits(buf: &mut Vec<u8>, state: &mut EncodeState, packed: u64, nbits: u32) {
    let mut byte_idx = 0u32;
    let mut bits_left = nbits;

    while bits_left >= 8 {
        encodebyte(buf, state, (packed >> (8 * byte_idx)) as u8);
        byte_idx += 1;
        bits_left -= 8;
    }
    if bits_left > 0 {
        let mask = (1u64 << bits_left) - 1;
        encodebits(buf, state, ((packed >> (8 * byte_idx)) & mask) as u32, bits_left as usize);
    }
}

/// Multiply byte array by a factor and propagate carry.
fn multiply_bytes(bytes: &mut [u8; 32], nbytes: &mut usize, factor: u64) {
    let mut carry = 0u64;
    for byte in bytes.iter_mut().take(*nbytes) {
        carry += *byte as u64 * factor;
        *byte = (carry & 0xff) as u8;
        carry >>= 8;
    }
    while carry > 0 {
        bytes[*nbytes] = (carry & 0xff) as u8;
        carry >>= 8;
        *nbytes += 1;
    }
}

/// Add a value to byte array and propagate carry.
fn add_to_bytes(bytes: &mut [u8; 32], nbytes: &mut usize, value: u64) {
    let mut carry = value;
    for byte in bytes.iter_mut().take((*nbytes).max(4)) {
        carry += *byte as u64;
        *byte = (carry & 0xff) as u8;
        carry >>= 8;
    }
    while carry > 0 {
        bytes[*nbytes] = (carry & 0xff) as u8;
        carry >>= 8;
        *nbytes += 1;
    }
}

fn encodeints(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    nbits: u32,
    sizes: [u32; 3],
    nums: [i32; 3],
) {
    if nbits <= 32 {
        write_packed_bits(buf, state, pack_into_u32(nums, sizes) as u64, nbits);
        return;
    }

    if nbits <= 64 {
        write_packed_bits(buf, state, pack_into_u64(nums, sizes), nbits);
        return;
    }

    // For very large nbits, use the byte array method (inverse of decodeints).
    let mut bytes = [0u8; 32];
    let mut nbytes = 0usize;

    // Initialize with nums[2].
    let mut carry = nums[2] as u32;
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (carry & 0xff) as u8;
        carry >>= 8;
        if carry == 0 {
            nbytes = i + 1;
            break;
        }
    }

    // Pack: result = ((nums[2] * sizes[2] + nums[1]) * sizes[1] + nums[0])
    multiply_bytes(&mut bytes, &mut nbytes, sizes[2] as u64);
    add_to_bytes(&mut bytes, &mut nbytes, nums[1] as u64);
    multiply_bytes(&mut bytes, &mut nbytes, sizes[1] as u64);
    add_to_bytes(&mut bytes, &mut nbytes, nums[0] as u64);

    // Write the bytes.
    let mut bits_left = nbits;
    let mut byte_idx = 0;
    while bits_left >= 8 {
        encodebyte(buf, state, bytes[byte_idx]);
        byte_idx += 1;
        bits_left -= 8;
    }
    if bits_left > 0 {
        encodebits(buf, state, bytes[byte_idx] as u32 & ((1 << bits_left) - 1), bits_left as usize);
    }
}

/// Returns the number of bits needed to represent `size`.
#[inline]
const fn sizeofint(size: u32) -> u32 {
    if size == 0 {
        0
    } else {
        u32::BITS - size.leading_zeros()
    }
}

/// Calculate total bits needed to represent the product of three sizes.
const fn sizeofints(sizes: [u32; 3]) -> u32 {
    let mut product_bytes = [0u8; 32];
    product_bytes[0] = 1;
    let mut byte_count = 1usize;

    // Multiply all sizes together in byte representation.
    let mut size_idx = 0;
    while size_idx < 3 {
        let size = sizes[size_idx];
        let mut carry = 0u32;
        let mut i = 0;
        while i < byte_count {
            carry += product_bytes[i] as u32 * size;
            product_bytes[i] = (carry & 0xff) as u8;
            carry >>= 8;
            i += 1;
        }
        while carry != 0 {
            product_bytes[byte_count] = (carry & 0xff) as u8;
            byte_count += 1;
            carry >>= 8;
        }
        size_idx += 1;
    }

    // Count bits in the most significant byte.
    let msb_index = byte_count - 1;
    let msb_bits = sizeofint(product_bytes[msb_index] as u32);

    msb_index as u32 * 8 + msb_bits
}

fn calc_sizeint(
    minint: [i32; 3],
    maxint: [i32; 3],
    sizeint: &mut [u32; 3],
    bitsizeint: &mut [u32; 3],
) -> u32 {
    for i in 0..3 {
        sizeint[i] = (maxint[i] - minint[i]) as u32 + 1;
    }
    bitsizeint.fill(0);

    // Check if any size is too large to multiply safely.
    let needs_separate_encoding = sizeint.iter().any(|&s| s > MAX_MULTIPLIABLE_SIZE);
    if needs_separate_encoding {
        for i in 0..3 {
            bitsizeint[i] = sizeofint(sizeint[i]);
        }
        return 0; // Signals separate encoding for each dimension.
    }

    sizeofints(*sizeint)
}

/// Write compressed positions to the writer.
/// Returns the number of compressed bytes written.
///
/// # Errors
/// Returns an error if writing to the underlying writer fails.
///
/// # Panics
/// Panics if `positions.len()` is not divisible by 3.
pub fn write_compressed_positions<W: Write>(
    writer: &mut W,
    positions: &[f32],
    precision: f32,
    magic: Magic,
) -> io::Result<usize> {
    assert_eq!(positions.len() % 3, 0);

    let to_int = |f: f32| (f * precision).round() as i32;
    let mut int_coords: Vec<[i32; 3]> = positions
        .chunks_exact(3)
        .map(|p| [to_int(p[0]), to_int(p[1]), to_int(p[2])])
        .collect();

    write_int_positions(writer, &mut int_coords, magic)
}

// Internal function that writes int positions
pub(crate) fn write_int_positions<W: Write>(
    writer: &mut W,
    int_coords: &mut [[i32;3]],
    magic: Magic,
) -> io::Result<usize> {
    let (minint, maxint) = calc_bounds(&int_coords);
    let smallidx = find_initial_smallidx(&int_coords);

    write_prelude(writer, &minint, &maxint, smallidx)?;

    let mut sizeint = [0u32; 3];
    let mut bitsizeint = [0u32; 3];
    let bitsize = calc_sizeint(minint, maxint, &mut sizeint, &mut bitsizeint);

    // Pre-allocate buffer to reduce reallocations during encoding.
    let natoms = int_coords.len();
    let mut compressed = Vec::with_capacity(natoms * 12);
    let mut state = EncodeState::default();

    encode_coordinates(
        &mut compressed,
        &mut state,
        int_coords,
        minint,
        bitsize,
        &sizeint,
        &bitsizeint,
        smallidx,
    );

    flush_bits(&mut compressed, &mut state);
    write_compressed_data(writer, &compressed, magic)?;

    Ok(compressed.len())
}


/// Encode coordinates with run-length compression and water swap.
#[allow(clippy::too_many_arguments)]
fn encode_coordinates(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    coords: &mut [[i32; 3]],
    minint: [i32; 3],
    bitsize: u32,
    sizeint: &[u32; 3],
    bitsizeint: &[u32; 3],
    mut smallidx: usize,
) {
    const LASTIDX: usize = MAGICINTS.len() - 1;
    let maxidx = LASTIDX.min(smallidx + 8);
    let minidx = maxidx.saturating_sub(8);

    let mut smaller = MAGICINTS[smallidx.saturating_sub(1).max(FIRSTIDX)] / 2;
    let mut small = MAGICINTS[smallidx] / 2;
    let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];
    let larger = MAGICINTS[maxidx] / 2;

    let mut idx = 0usize;
    let mut prevrun = 0usize;
    let mut first_run = true;
    let mut prevcoord = [0; 3];

    while idx < coords.len() {
        // Determine if we should adjust encoding precision.
        let mut size_change = if idx >= 1 {
            if smallidx < maxidx && coords_within_threshold(coords[idx], prevcoord, larger) {
                SizeChange::Increase
            } else if smallidx > minidx {
                SizeChange::Decrease
            } else {
                SizeChange::Same
            }
        } else {
            SizeChange::Same
        };

        // Water swap: if next coord is close, swap to improve compression.
        let mut can_run = idx + 1 < coords.len()
            && coords_within_threshold(coords[idx], coords[idx + 1], small);
        if can_run {
            coords.swap(idx, idx + 1);
        }

        // Encode the current coordinate.
        let coord = coords[idx];
        encode_full_coord(buf, state, coord, minint, bitsize, sizeint, bitsizeint);
        prevcoord = coord;
        idx += 1;

        // If not starting a run, don't decrease precision.
        if !can_run && size_change == SizeChange::Decrease {
            size_change = SizeChange::Same;
        }

        // Collect run-length encoded deltas.
        let mut run_deltas = [0i32; MAX_RUN_COORDS];
        let mut run = 0usize;

        while can_run && run < MAX_RUN_COORDS && idx < coords.len() {
            let next = coords[idx];

            // Check if delta is small enough to keep decreasing precision.
            if size_change == SizeChange::Decrease {
                let delta = [
                    next[0] - prevcoord[0],
                    next[1] - prevcoord[1],
                    next[2] - prevcoord[2],
                ];
                let dist_sq = delta.iter().map(|&d| (d as i64) * (d as i64)).sum::<i64>();
                if dist_sq >= (smaller as i64) * (smaller as i64) {
                    size_change = SizeChange::Same;
                }
            }

            // Store delta with offset.
            run_deltas[run] = next[0] - prevcoord[0] + small;
            run_deltas[run + 1] = next[1] - prevcoord[1] + small;
            run_deltas[run + 2] = next[2] - prevcoord[2] + small;
            run += 3;
            prevcoord = next;
            idx += 1;

            // Check if we can continue the run.
            can_run = idx < coords.len()
                && coords_within_threshold(coords[idx], prevcoord, small);
        }

        // Encode run header.
        let run_changed = first_run || run != prevrun || size_change != SizeChange::Same;
        first_run = false;

        if run_changed {
            prevrun = run;
            encodebits(buf, state, 1, 1);
            let size_delta: i32 = match size_change {
                SizeChange::Decrease => -1,
                SizeChange::Same => 0,
                SizeChange::Increase => 1,
            };
            let run_value = (run as i32 + size_delta + 1) as u32;
            encodebits(buf, state, run_value, 5);
        } else {
            encodebits(buf, state, 0, 1);
        }

        // Encode run deltas.
        for chunk in run_deltas[..run].chunks_exact(3) {
            encodeints(buf, state, smallidx as u32, sizesmall, [chunk[0], chunk[1], chunk[2]]);
        }

        // Adjust precision for next iteration.
        match size_change {
            SizeChange::Decrease => {
                smallidx = smallidx.saturating_sub(1);
                small = smaller;
                smaller = if smallidx > FIRSTIDX {
                    MAGICINTS[smallidx - 1] / 2
                } else {
                    0
                };
                sizesmall.fill(MAGICINTS[smallidx] as u32);
            }
            SizeChange::Increase => {
                smallidx = (smallidx + 1).min(LASTIDX);
                smaller = small;
                small = MAGICINTS[smallidx] / 2;
                sizesmall.fill(MAGICINTS[smallidx] as u32);
            }
            SizeChange::Same => {}
        }
    }
}

fn calc_bounds(int_coords: &[[i32; 3]]) -> ([i32; 3], [i32; 3]) {
    int_coords.iter().fold(
        ([i32::MAX; 3], [i32::MIN; 3]),
        |(mut min, mut max), coord| {
            for (i, &c) in coord.iter().enumerate() {
                min[i] = min[i].min(c);
                max[i] = max[i].max(c);
            }
            (min, max)
        },
    )
}

/// Find initial smallidx based on minimum delta between adjacent coordinates.
/// This matches the original C implementation which uses mindiff (sum of absolute
/// differences between consecutive coordinate triplets).
fn find_initial_smallidx(int_coords: &[[i32; 3]]) -> usize {
    let mindiff = int_coords
        .windows(2)
        .map(|w| {
            (w[0][0] - w[1][0]).abs() + (w[0][1] - w[1][1]).abs() + (w[0][2] - w[1][2]).abs()
        })
        .min()
        .unwrap_or(0);

    // Find first index where MAGICINTS[i] >= mindiff.
    MAGICINTS[FIRSTIDX..]
        .iter()
        .position(|&m| m >= mindiff)
        .map_or(MAGICINTS.len() - 1, |pos| FIRSTIDX + pos)
}

fn write_prelude<W: Write>(
    writer: &mut W,
    minint: &[i32; 3],
    maxint: &[i32; 3],
    smallidx: usize,
) -> io::Result<()> {
    for &v in minint.iter().chain(maxint) {
        writer.write_all(&v.to_be_bytes())?;
    }
    writer.write_all(&(smallidx as u32).to_be_bytes())
}

fn encode_full_coord(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    coord: [i32; 3],
    minint: [i32; 3],
    bitsize: u32,
    sizeint: &[u32; 3],
    bitsizeint: &[u32; 3],
) {
    let relative = [
        coord[0] - minint[0],
        coord[1] - minint[1],
        coord[2] - minint[2],
    ];
    if bitsize == 0 {
        encodebits(buf, state, relative[0] as u32, bitsizeint[0] as usize);
        encodebits(buf, state, relative[1] as u32, bitsizeint[1] as usize);
        encodebits(buf, state, relative[2] as u32, bitsizeint[2] as usize);
    } else {
        encodeints(buf, state, bitsize, *sizeint, relative);
    }
}

fn write_compressed_data<W: Write>(
    writer: &mut W,
    compressed: &[u8],
    magic: Magic,
) -> io::Result<()> {
    let nbytes = compressed.len();
    match magic {
        Magic::Xtc1995 => writer.write_all(&(nbytes as u32).to_be_bytes())?,
        Magic::Xtc2023 => writer.write_all(&(nbytes as u64).to_be_bytes())?,
    }
    writer.write_all(compressed)?;
    let pad = padding(nbytes);
    if pad > 0 {
        writer.write_all(&ZERO_PAD[..pad])?;
    }
    Ok(())
}
