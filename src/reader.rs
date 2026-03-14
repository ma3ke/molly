use std::io::{self, Read};

use crate::buffer::Buffered;
use crate::selection::AtomSelection;
use crate::{BoxVec, Magic};

struct DecodeState {
    lastbits: usize,
    lastbyte: u8,
}

// TODO: I have a constexpr laying around for this somewhere.
#[rustfmt::skip]
pub const MAGICINTS: [i32; 73] = [
    0,        0,        0,       0,       0,       0,       0,       0,       0,       8,
    10,       12,       16,      20,      25,      32,      40,      50,      64,      80,
    101,      128,      161,     203,     256,     322,     406,     512,     645,     812,
    1024,     1290,     1625,    2048,    2580,    3250,    4096,    5060,    6501,    8192,
    10321,    13003,    16384,   20642,   26007,   32768,   41285,   52015,   65536,   82570,
    104031,   131072,   165140,  208063,  262144,  330280,  416127,  524287,  660561,  832255,
    1048576,  1321122,  1664510, 2097152, 2642245, 3329021, 4194304, 5284491, 6658042, 8388607,
    10568983, 13316085, 16777216
];
pub const FIRSTIDX: usize = 9; // Note that MAGICINTS[FIRSTIDX-1] == 0.

/// The number of bytes that together form the prelude of `maxint`, `minint`, and `smallidx`.
pub const NBYTES_POSITIONS_PRELUDE: usize = 7 * 4;

#[inline]
/// The low-level decompression routine.
///
/// If successful, returns the number of compressed bytes that were read.
///
/// `header_natoms` must be greater than or equal to the number of `positions`.
pub fn read_compressed_positions<'s, 'r, B: Buffered<'s, 'r, R>, R: Read>(
    file: &'r mut R,
    header_natoms: usize,
    positions: &mut [f32],
    precision: f32,
    scratch: &'s mut Vec<u8>,
    atom_selection: &AtomSelection,
    magic: Magic,
) -> io::Result<usize> {
    let natoms_out = {
        let n = positions.len();
        assert_eq!(n % 3, 0, "the length of `positions` must be divisible by 3");
        n / 3
    };

    let nselected = atom_selection.natoms_selected(header_natoms);
    if nselected < natoms_out {
        let ninvalid = natoms_out - nselected;
        eprintln!(
            "WARNING [molly {}:{}]: The atoms selection includes {nselected} atoms, but the positions \
            buffer has {natoms_out} spots. This means that {ninvalid} trailing coordinates in the \
            positions buffer will be invalid.",
            file!(),
            line!()
        )
    }

    let invprecision = precision.recip();

    // TODO: Once `array_try_map` is stable, both of these inits can be cleaned up significantly.
    let minint = [0; 3]
        .map(|_| read_i32(file))
        .into_iter()
        .collect::<io::Result<Vec<_>>>()?
        .try_into()
        .unwrap();
    let maxint = [0; 3]
        .map(|_| read_i32(file))
        .into_iter()
        .collect::<io::Result<Vec<_>>>()?
        .try_into()
        .unwrap();
    let smallidx = read_u32(file)?;
    assert_eq!(
        std::mem::size_of_val(&minint)
            + std::mem::size_of_val(&maxint)
            + std::mem::size_of_val(&smallidx),
        NBYTES_POSITIONS_PRELUDE
    );
    let mut smallidx = smallidx as usize;
    assert!(smallidx < MAGICINTS.len());

    let mut sizeint = [0u32; 3];
    let mut bitsizeint = [0u32; 3];
    let bitsize = calc_sizeint(minint, maxint, &mut sizeint, &mut bitsizeint);

    let tmpidx = smallidx - 1;
    let tmpidx = if FIRSTIDX > tmpidx { FIRSTIDX } else { tmpidx };

    let mut smaller = MAGICINTS[tmpidx] / 2;
    let mut smallnum = MAGICINTS[smallidx] / 2;
    let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];

    scratch.clear();
    let mut buffer = B::new(scratch, file, magic)?;

    let mut state = DecodeState {
        lastbits: 0,
        lastbyte: 0,
    };
    let mut run: i32 = 0;
    let mut prevcoord;
    let mut write_idx = 0;
    let mut read_idx = 0;
    // The number of positions to be read to fulfill an AtomSelection may not be equal to natoms!
    assert!(header_natoms >= natoms_out);
    let limit = atom_selection.reading_limit(header_natoms);
    'decompress: while read_idx < limit {
        let mut coord = [0i32; 3];
        let Some(mut position) = positions
            .chunks_exact_mut(3)
            .nth(write_idx)
            .map(|pos| -> &mut [f32; 3] { pos.try_into().unwrap() })
        else {
            break 'decompress;
        };
        if bitsize == 0 {
            coord[0] = decodebits::<_, R>(&mut buffer, &mut state, bitsizeint[0] as usize);
            coord[1] = decodebits::<_, R>(&mut buffer, &mut state, bitsizeint[1] as usize);
            coord[2] = decodebits::<_, R>(&mut buffer, &mut state, bitsizeint[2] as usize);
        } else {
            decodeints::<R>(&mut buffer, &mut state, bitsize, sizeint, &mut coord);
        }

        coord[0] += minint[0];
        coord[1] += minint[1];
        coord[2] += minint[2];
        prevcoord = coord;

        macro_rules! write_position {
            ($position:ident, $write_idx:ident, $read_idx:ident, $coord:ident) => {
                let is_included = atom_selection.is_included($read_idx);
                $read_idx += 1;
                match is_included {
                    None => break 'decompress,
                    Some(false) => {}
                    Some(true) => {
                        *$position = $coord.map(|v| v as f32 * invprecision);
                        $write_idx += 1;
                    }
                }
                if read_idx >= limit {
                    // This additional break is necessary because under some conditions within the
                    // inner loop that deals with runs we may end up trying to pop bytes from the
                    // buffer that don't actually exist, because we're trying to read a coordinate with
                    // an index beyond the encoded range.
                    break 'decompress;
                }
            };
        }

        let flag: bool = decodebits::<u8, R>(&mut buffer, &mut state, 1) > 0;
        let mut is_smaller = 0;
        if flag {
            run = decodebits::<_, R>(&mut buffer, &mut state, 5);
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            // TODO: Investigate whether this is something we can just remove. I believe it may be.
            // if write_idx * 3 + run as usize > n {
            //     eprintln!("may attempt to write a run beyond the positions buffer");
            //     dbg!(write_idx, run, n, write_idx * 3 + run as usize);
            // }

            // Let's read the next coordinate.
            coord.fill(0);

            for k in (0..run).step_by(3) {
                decodeints::<R>(
                    &mut buffer,
                    &mut state,
                    smallidx as u32,
                    sizesmall,
                    &mut coord,
                );
                // let mut current_coord_read_idx = read_idx;
                // read_idx += 1;
                coord[0] += prevcoord[0] - smallnum;
                coord[1] += prevcoord[1] - smallnum;
                coord[2] += prevcoord[2] - smallnum;
                if k == 0 {
                    // Swap the first and second atom. This is done to achieve better compression
                    // for water atoms. Waters are stored as OHH, but right now we want to swap the
                    // atoms such that e.g., water will become HOH again.
                    std::mem::swap(&mut coord, &mut prevcoord);
                    write_position!(position, write_idx, read_idx, prevcoord);
                    position = match positions.chunks_exact_mut(3).nth(write_idx) {
                        Some(c) => c.try_into().unwrap(),
                        None => break 'decompress,
                    };
                } else {
                    prevcoord = coord;
                }
                write_position!(position, write_idx, read_idx, coord);
                position = match positions.chunks_exact_mut(3).nth(write_idx) {
                    Some(c) => c.try_into().unwrap(),
                    None => break 'decompress,
                };
            }
        } else {
            write_position!(position, write_idx, read_idx, coord);
        }

        match is_smaller.cmp(&0) {
            std::cmp::Ordering::Less => {
                smallidx -= 1;
                smallnum = smaller;
                if smallidx > FIRSTIDX {
                    smaller = MAGICINTS[smallidx - 1] / 2;
                } else {
                    smaller = 0;
                }
            }
            std::cmp::Ordering::Greater => {
                smallidx += 1;
                smaller = smallnum;
                smallnum = MAGICINTS[smallidx] / 2;
            }
            std::cmp::Ordering::Equal => {}
        }

        assert_ne!(MAGICINTS[smallidx], 0, "found an invalid size");
        sizesmall.fill(MAGICINTS[smallidx] as u32);
    }

    if write_idx < natoms_out {
        eprintln!(
            "WARNING [molly {}:{}]: Could not fill entire positions buffer \
            (write_idx = {write_idx}, natoms_out = {natoms_out})",
            file!(),
            line!()
        )
    }

    // The number of bytes that were read during decompression.
    let nbytes = buffer.tell();
    buffer.finish()?;

    Ok(nbytes)
}

#[inline]
pub(crate) fn read_boxvec<R: Read>(file: &mut R) -> io::Result<BoxVec> {
    let mut boxvec = [0.0; 9];
    read_f32s(file, &mut boxvec)?;
    #[rustfmt::skip]
    let boxvec = [
        boxvec[0], boxvec[1], boxvec[2],
        boxvec[3], boxvec[4], boxvec[5],
        boxvec[6], boxvec[7], boxvec[8],
    ];
    Ok(boxvec)
}

pub(crate) fn read_f32s<R: Read>(file: &mut R, buf: &mut [f32]) -> io::Result<()> {
    for value in buf {
        *value = read_f32(file)?
    }
    Ok(())
}

// FIXME: These read_* functions are prime targets for a macro tbh.
pub(crate) fn read_f32<R: Read>(file: &mut R) -> io::Result<f32> {
    let mut buf: [u8; 4] = Default::default();
    file.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}

pub(crate) fn read_i32<R: Read>(file: &mut R) -> io::Result<i32> {
    let mut buf: [u8; 4] = Default::default();
    file.read_exact(&mut buf)?;
    Ok(i32::from_be_bytes(buf))
}

pub(crate) fn read_u32<R: Read>(file: &mut R) -> io::Result<u32> {
    let mut buf: [u8; 4] = Default::default();
    file.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub(crate) fn read_u64<R: Read>(file: &mut R) -> io::Result<u64> {
    let mut buf: [u8; 8] = Default::default();
    file.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

pub(crate) fn read_f64<R: Read>(file: &mut R) -> io::Result<f64> {
    let mut buf: [u8; 8] = Default::default();
    file.read_exact(&mut buf)?;
    Ok(f64::from_be_bytes(buf))
}

pub fn read_nbytes<R: Read>(reader: &mut R, magic: Magic) -> io::Result<usize> {
    let nbytes = match magic {
        Magic::Xtc1995 => read_u32(reader)? as usize,
        Magic::Xtc2023 => read_u64(reader)? as usize,
    };
    Ok(nbytes)
}

fn calc_sizeint(
    minint: [i32; 3],
    maxint: [i32; 3],
    sizeint: &mut [u32; 3],
    bitsizeint: &mut [u32; 3],
) -> u32 {
    sizeint[0] = (maxint[0] - minint[0]) as u32 + 1;
    sizeint[1] = (maxint[1] - minint[1]) as u32 + 1;
    sizeint[2] = (maxint[2] - minint[2]) as u32 + 1;

    bitsizeint.fill(0);

    // Check if one of the sizes is too big to be multiplied.
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        return 0; // This flags the use of large sizes. // FIXME: This can become an enum to be more explicit?
    }

    sizeofints(*sizeint)
}

#[inline]
const fn sizeofint(size: u32) -> u32 {
    let mut n = 1;
    let mut nbits = 0;

    while size >= n && nbits < 32 {
        nbits += 1;
        n <<= 1;
    }

    nbits
}

fn sizeofints(sizes: [u32; 3]) -> u32 {
    let mut nbytes = 1;
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let mut nbits = 0;

    for size in sizes {
        let mut tmp = 0;
        let mut bytecount = 0;
        while bytecount < nbytes {
            tmp += bytes[bytecount] as u32 * size;
            bytes[bytecount] = (tmp & 0xff) as u8;
            tmp >>= 8;
            bytecount += 1;
        }
        while tmp != 0 {
            bytes[bytecount] = (tmp & 0xff) as u8;
            bytecount += 1;
            tmp >>= 8;
        }
        nbytes = bytecount;
    }

    nbytes -= 1;
    let mut num = 1;
    while bytes[nbytes] as u32 >= num {
        nbits += 1;
        num *= 2;
    }

    nbytes as u32 * 8 + nbits // FIXME: Check whether it is okay for nbytes to have the type of usize not u32
}

fn decodebyte<'s, 'r, R>(buf: &mut impl Buffered<'s, 'r, R>, state: &mut DecodeState) -> u8 {
    let mask = 0xff;

    let DecodeState {
        mut lastbits,
        lastbyte,
    } = *state;
    let mut lastbyte = lastbyte as u32;

    let mut num = 0;
    let mut nbits = 8;
    while nbits >= 8 {
        lastbyte = (lastbyte << 8) | buf.pop() as u32;
        num |= (lastbyte >> lastbits) << (nbits - 8);
        nbits -= 8;
    }

    if nbits > 0 {
        if lastbits < nbits {
            lastbits += 8;
            lastbyte = (lastbyte << 8) | buf.pop() as u32;
        }
        lastbits -= nbits;
        num |= (lastbyte >> lastbits) & mask;
    }

    num &= mask;
    *state = DecodeState {
        lastbits,
        lastbyte: (lastbyte & 0xff) as u8, // We don't care about anything but the last byte.
    };

    debug_assert_eq!(num & 0xff, num);
    num as u8
}

fn decodebits<'s, 'r, T: TryFrom<u32>, R: Read>(
    buf: &mut impl Buffered<'s, 'r, R>,
    state: &mut DecodeState,
    mut nbits: usize,
) -> T {
    let mask = (1 << nbits) - 1; // A string of ones that is nbits long.

    let DecodeState {
        mut lastbits,
        lastbyte,
    } = *state;
    let mut lastbyte = lastbyte as u32;

    let mut num = 0;
    while nbits >= 8 {
        lastbyte = (lastbyte << 8) | buf.pop() as u32;
        num |= (lastbyte >> lastbits) << (nbits - 8);
        nbits -= 8;
    }

    if nbits > 0 {
        if lastbits < nbits {
            lastbits += 8;
            lastbyte = (lastbyte << 8) | buf.pop() as u32;
        }
        lastbits -= nbits;
        num |= (lastbyte >> lastbits) & mask;
    }

    num &= mask;
    *state = DecodeState {
        lastbits,
        lastbyte: (lastbyte & 0xff) as u8, // We don't care about anything but the last byte.
    };

    match num.try_into() {
        Ok(n) => n,
        Err(_) => unreachable!(), // We just checked for that!
    }
}

fn decodeints<'s, 'r, R: Read>(
    buf: &mut impl Buffered<'s, 'r, R>,
    state: &mut DecodeState,
    mut nbits: u32,
    sizes: [u32; 3],
    nums: &mut [i32; 3],
) {
    if nbits <= 32 {
        unpack_from_int_into_u32(buf, state, nbits, sizes, nums);
        return;
    }
    if nbits <= 64 {
        unpack_from_int_into_u64(buf, state, nbits, sizes, nums);
        return;
    }

    let mut bytes = [0u8; 32];
    let mut nbytes: usize = 0;
    while nbits >= 8 {
        bytes[nbytes] = decodebyte(buf, state);
        nbytes += 1;
        nbits -= 8;
    }
    if nbits > 0 {
        bytes[nbytes] = decodebits(buf, state, nbits as usize);
        nbytes += 1;
    }

    for i in (1..=2).rev() {
        let mut num: u32 = 0;
        for j in 0..nbytes {
            let k = nbytes - 1 - j;
            num = (num << 8) | bytes[k] as u32;
            let p = num / sizes[i];
            bytes[k] = p as u8;
            num -= p * sizes[i];
        }
        nums[i] = num as i32;
    }

    nums[0] = i32::from_le_bytes(bytes[..4].try_into().unwrap());
}

fn unpack_from_int_into_u32<'s, 'r, R: Read>(
    buf: &mut impl Buffered<'s, 'r, R>,
    state: &mut DecodeState,
    mut nbits: u32,
    sizes: [u32; 3],
    nums: &mut [i32; 3],
) {
    type T = u32;
    let mut v: T = 0;
    let mut nbytes: usize = 0;
    while nbits >= 8 {
        let byte: T = decodebyte(buf, state) as T;
        v |= byte << (8 * nbytes as u32);
        nbytes += 1;
        nbits -= 8;
    }
    if nbits > 0 {
        let byte: T = decodebits(buf, state, nbits as usize);
        v |= byte << (8 * nbytes as u32);
    }

    // FIXME: What's up with the whole FastType stuff here?
    let sz: T = sizes[2];
    let sy: T = sizes[1];
    let szy: T = sz * sy;
    let x1 = v / szy;
    let q1 = v - x1 * szy;
    let y1 = q1 / sz;
    let z1 = q1 - y1 * sz;

    *nums = [x1, y1, z1].map(|v| v as i32);
}

fn unpack_from_int_into_u64<'s, 'r, R: Read>(
    buf: &mut impl Buffered<'s, 'r, R>,
    state: &mut DecodeState,
    mut nbits: u32,
    sizes: [u32; 3],
    nums: &mut [i32; 3],
) {
    type T = u64;
    let mut v: T = 0;
    let mut nbytes: usize = 0;
    while nbits >= 8 {
        let byte: T = decodebyte(buf, state) as T;
        v |= byte << (8 * nbytes as u32);
        nbytes += 1;
        nbits -= 8;
    }
    if nbits > 0 {
        let byte: T = decodebits(buf, state, nbits as usize);
        v |= byte << (8 * nbytes as u32);
    }

    // FIXME: What's up with the whole FastType stuff here?
    let sz: T = sizes[2] as u64;
    let sy: T = sizes[1] as u64;
    let szy: T = sz * sy;
    let x1 = v / szy;
    let q1 = v - x1 * szy;
    let y1 = q1 / sz;
    let z1 = q1 - y1 * sz;

    *nums = [x1, y1, z1].map(|v| v as i32);
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Seek};

    use super::*;
    use crate::buffer::{Buffer, UnBuffered};

    const HEADER_BYTES: usize = 60;
    const N_ATOMS: usize = 125;
    #[rustfmt::skip]
    #[allow(clippy::excessive_precision)]
    const CORRECT_POSITIONS: [f32; 375] = [
        0.86700004, 1.24200010, 0.83700001,  0.84400004, 1.25100004, 0.85100001,
        0.89300006, 1.22100007, 0.83700001,  0.88800007, 1.19900000, 0.85200005,
        0.90600001, 1.22300004, 0.80800002,  0.93800002, 1.23500001, 0.81400001,
        0.90800005, 1.23100006, 0.77400004,  0.89200001, 1.21400010, 0.75700002,
        0.94000005, 1.22800004, 0.75800001,  0.96400004, 1.20900011, 0.75300002,
        0.98600006, 1.21300005, 0.76600003,  0.96000003, 1.18100011, 0.73800003,
        0.93300002, 1.18600010, 0.72500002,  0.90700006, 1.17800009, 0.72300004,
        0.91900002, 1.19100010, 0.70300000,  0.96800005, 1.15800011, 0.75500005,
        1.00100004, 1.15500009, 0.75800001,  0.93800002, 1.14200007, 0.76500004,
        0.93900007, 1.14100003, 0.79900002,  0.92500007, 1.13300001, 0.73600006,
        0.89600002, 1.12999999, 0.73300004,  0.94100004, 1.13400006, 0.70900005,
        0.92500007, 1.15000009, 0.68800002,  0.91700005, 1.13900005, 0.66500002,
        0.92100006, 1.16500008, 0.66600000,  0.97400003, 1.13300001, 0.71300005,
        0.99800002, 1.14500010, 0.69400000,  1.01500010, 1.13300001, 0.66900002,
        0.98500007, 1.11000001, 0.73000001,  1.01700007, 1.11000001, 0.72600001,
        1.04100000, 1.10300004, 0.72000002,  0.96400004, 1.08800005, 0.72400003,
        0.94900006, 1.08300006, 0.74900001,  0.96700006, 1.08800005, 0.69300001,
        0.94100004, 1.08800005, 0.67800003,  0.92300003, 1.07300007, 0.66200006,
        0.99800002, 1.07700002, 0.69200003,  1.00999999, 1.08700001, 0.66100001,
        0.99500006, 1.04900002, 0.70000004,  1.00499999, 1.04900002, 0.73200005,
        0.98700004, 1.03600001, 0.67200005,  0.97100007, 1.05300009, 0.65500003,
        0.96900004, 1.07400000, 0.63800001,  0.95800006, 1.05000007, 0.63100004,
        1.01500010, 1.03900003, 0.65200001,  1.00600004, 1.02300000, 0.63000005,
        1.03800010, 1.03600001, 0.67500001,  1.05099999, 1.06100010, 0.68900001,
        1.05400002, 1.00700008, 0.67100000,  1.05000007, 1.00400006, 0.63300001,
        1.08300006, 1.00400006, 0.66900002,  1.08200001, 1.02700006, 0.70600003,
        1.10600006, 1.01700007, 0.66400003,  1.08500003, 1.05099999, 0.65600001,
        1.13100004, 1.00999999, 0.66300004,  1.15200006, 1.02000010, 0.68500006,
        1.17599999, 1.01100003, 0.69400000,  1.14100003, 1.04500007, 0.70000004,
        1.11500000, 1.05800008, 0.69200003,  1.09600007, 1.07700002, 0.69600003,
        1.16000008, 1.05800008, 0.72100001,  1.19000005, 1.04600000, 0.72300004,
        1.15000009, 1.08900010, 0.73700004,  1.12700009, 1.10000002, 0.73000001,
        1.17400002, 1.11000001, 0.75000005,  1.19200003, 1.09300005, 0.75900006,
        1.18500006, 1.14100003, 0.76100003,  1.15600001, 1.15400004, 0.76800000,
        1.20700001, 1.16100001, 0.77200001,  1.23400008, 1.14800000, 0.78500002,
        1.23500001, 1.12200009, 0.79200005,  1.21600008, 1.19300007, 0.78100001,
        1.23300004, 1.20100009, 0.76100003,  1.23000001, 1.19300007, 0.73400002,
        1.24100005, 1.24300003, 0.75900006,  1.26200008, 1.25000000, 0.75200003,
        1.22100007, 1.26000010, 0.77400004,  1.24500000, 1.26200008, 0.80000001,
        1.19300007, 1.25000000, 0.76100003,  1.17599999, 1.27900004, 0.75400000,
        1.16400003, 1.29900002, 0.76800000,  1.16500008, 1.30000007, 0.74200004,
        1.19600009, 1.22700011, 0.74500006,  1.20700001, 1.23500001, 0.72000002,
        1.23000001, 1.24300003, 0.70800000,  1.21100008, 1.23000001, 0.69300001,
        1.18000006, 1.19600009, 0.74000000,  1.16800010, 1.20000004, 0.77500003,
        1.18600010, 1.17200005, 0.71900004,  1.19800007, 1.18100011, 0.69700002,
        1.17700004, 1.14000010, 0.71200001,  1.14500010, 1.14400005, 0.71600002,
        1.19600009, 1.10699999, 0.70600003,  1.21800005, 1.11800003, 0.72200006,
        1.23700010, 1.11600005, 0.74100005,  1.23000001, 1.14000010, 0.73200005,
        1.17599999, 1.08600008, 0.68100005,  1.15299999, 1.09600007, 0.68700003,
        1.18700003, 1.05400002, 0.67200005,  1.21100008, 1.05099999, 0.68200004,
        1.17599999, 1.04300010, 0.65100002,  1.16000008, 1.03100001, 0.62700003,
        1.13900005, 1.00800001, 0.61300003,  1.12700009, 1.05099999, 0.62900000,
        1.13200008, 1.06800007, 0.64900004,  1.10500001, 1.04100000, 0.60300004,
        1.08200001, 1.03400003, 0.61100000,  1.10699999, 1.04400002, 0.56800001,
        1.08500003, 1.02600002, 0.55400002,  1.13500010, 1.04700005, 0.56100004,
        1.15500009, 1.03400003, 0.57600003,  1.16600000, 1.00600004, 0.55000001,
        1.18400001, 1.05000007, 0.58400005,  1.20400011, 1.04100000, 0.57200002,
        1.17200005, 1.07300007, 0.60600000,  1.14700007, 1.08100008, 0.59600001,
        1.17900002, 1.09400010, 0.63100004,  1.21100008, 1.10100007, 0.64100003,
        1.22300004, 1.09900009, 0.66600000,  1.23500001, 1.11200010, 0.64600002,
        1.16300010, 1.11500000, 0.63600003,  1.13900005, 1.12500000, 0.61900001,
        1.11400008, 1.13400006, 0.61200004,  1.13300001, 1.13200008, 0.59300005,
        1.16400003, 1.14300000, 0.65800005,
    ];

    // TODO: Add a set of tests for the 2023 magic number as well. In order to do that, we need a
    // 2023-magic number counterpart to 'delinyah_tiny.xtc'.
    mod magic_1995 {
        use super::*;
        const MAGIC: Magic = Magic::Xtc1995;

        #[test]
        fn read_compressed() -> std::io::Result<()> {
            // A hand-tweaked test frame, derived from `delinyah_smaller.xtc`. Describes 125 positions.
            let bytes = include_bytes!("../tests/trajectories/delinyah_tiny.xtc");
            let position_bytes = &bytes[HEADER_BYTES..]; // Skip the header.

            let mut positions = vec![0.0; N_ATOMS * 3];
            let mut scratch = Vec::new();
            let precision = 1000.0;
            let mut data = BufReader::new(position_bytes);
            read_compressed_positions::<UnBuffered, _>(
                &mut data,
                N_ATOMS,
                &mut positions,
                precision,
                &mut scratch,
                &AtomSelection::Until(N_ATOMS as u32),
                MAGIC,
            )?;

            assert_eq!(positions.len(), N_ATOMS * 3); // We know this but still.
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            assert_eq!(positions, CORRECT_POSITIONS);

            Ok(())
        }

        #[test]
        fn read_compressed_from_file() -> std::io::Result<()> {
            // A hand-tweaked test frame, derived from `delinyah_smaller.xtc`. Describes 125 positions.
            let mut file = std::fs::File::open("tests/trajectories/delinyah_tiny.xtc")?;
            file.seek(io::SeekFrom::Start(HEADER_BYTES as u64))?; // Skip the header.

            let mut positions = vec![0.0; N_ATOMS * 3];
            let mut scratch = Vec::new();
            let precision = 1000.0;
            read_compressed_positions::<Buffer, _>(
                &mut file,
                N_ATOMS,
                &mut positions,
                precision,
                &mut scratch,
                &AtomSelection::Until(N_ATOMS as u32),
                MAGIC,
            )?;

            assert_eq!(positions.len(), N_ATOMS * 3); // We know this but still.
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            assert_eq!(positions, CORRECT_POSITIONS);

            Ok(())
        }

        #[test]
        fn read_compressed_atom_selection_list() -> std::io::Result<()> {
            // A hand-tweaked test frame, derived from `delinyah_smaller.xtc`. Describes 125 positions.
            let bytes = include_bytes!("../tests/trajectories/delinyah_tiny.xtc");
            let position_bytes = &bytes[HEADER_BYTES..]; // Skip the header.
            let mut scratch = Vec::new();
            let precision = 1000.0;

            let mut read_positions = |selection| -> std::io::Result<_> {
                let mut positions = vec![f32::NAN; N_ATOMS * 3];
                let mut data = BufReader::new(position_bytes);
                read_compressed_positions::<UnBuffered, _>(
                    &mut data,
                    N_ATOMS,
                    &mut positions,
                    precision,
                    &mut scratch,
                    &selection,
                    MAGIC,
                )?;
                Ok(positions)
            };

            // All positions should be correct.
            let selection = AtomSelection::Mask(vec![true; N_ATOMS]);
            let positions = read_positions(selection)?;
            assert_eq!(positions.len(), N_ATOMS * 3); // We know this but still.
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            assert_eq!(positions, CORRECT_POSITIONS);

            // All positions should be NaN, since the selection is empty.
            let selection = AtomSelection::Mask(vec![false; N_ATOMS]);
            let positions = read_positions(selection)?;
            assert_eq!(positions.len(), N_ATOMS * 3);
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            assert!(positions.into_iter().all(f32::is_nan));

            // With the interleaved selection, we expect a correct position followed by a NaN,
            // repeated.
            let interleaved = Vec::from_iter((0..N_ATOMS as u32).map(|i| i % 2 == 0));
            let selection = AtomSelection::Mask(interleaved);
            let positions = read_positions(selection)?;
            assert_eq!(positions.len(), N_ATOMS * 3);
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            for (i, (position, correct)) in positions
                .chunks_exact(3)
                .zip(CORRECT_POSITIONS.chunks_exact(3).step_by(2))
                .enumerate()
            {
                if i < 63 {
                    // Expect a correct position.
                    assert_eq!(position, correct);
                } else {
                    // Expect an unfilled position, marked by NaNs.
                    assert_eq!(position, [f32::NAN; 3]);
                }
            }

            // Now, let's try a pattern of falses, trues, falses, trues, and finally falses.
            let mask: Vec<_> = vec![
                vec![false; 12],
                vec![true; 32],
                vec![false; 56],
                vec![true; 11],
                vec![false; 14],
            ]
            .into_iter()
            .flatten()
            .collect();
            let selection = AtomSelection::Mask(mask.clone());
            let positions = read_positions(selection)?;
            assert_eq!(positions.len(), N_ATOMS * 3);
            assert_eq!(positions.len(), CORRECT_POSITIONS.len());
            let n_nans_expected = mask.iter().filter(|&&v| !v).count() * 3;
            let n_nans = positions.iter().filter(|&&v| v.is_nan()).count();
            assert_eq!(n_nans, n_nans_expected);
            let mut positions = positions.chunks_exact(3);
            let mut corrects = CORRECT_POSITIONS.chunks_exact(3);
            for selected in mask {
                let correct = corrects.next().unwrap();
                if selected {
                    // Expect a correct position.
                    let position = positions.next().unwrap();
                    assert_eq!(position, correct);
                }
            }
            assert!(positions.clone().flatten().all(|v| v.is_nan()));
            assert_eq!(positions.count() * 3, n_nans_expected);

            Ok(())
        }
    }
}
