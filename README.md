# _molly_&mdash;read `xtc` files, fast

> **NOTE:**
> This repository is hosted on [sourcehut][sourcehut] and recently on
> [GitHub][github] as well.

![molly logo](molly.svg)

A reader and writer for the Gromacs [xtc file format][xtc] implemented in pure Rust.

_molly_ tries to decompress and read the minimal number of bytes from disk. To
this end, the library features extensive selection methods for **frames**
within a trajectory and **atoms** within each frame. This selection allows for
some exciting optimizations&mdash;only the necessary positions are
decompressed.
Similarly, there are cases where only a limited number of compressed bytes are
read in the first place. This is particularly powerful in applications where a
subset of positions at the top-end of the frame is selected in a large
trajectory.
Such buffered reading can be very beneficial when disk read speed is
particularly poor, such as over networked file storage.

_molly_ also supports **writing** XTC files, enabling roundtrip workflows where
trajectories can be read, processed, and written back out.

For convenient use in existing analysis tools, _molly_ exposes a set of
bindings that allow access to its functions from Python.

_molly_ can also be installed as a command line tool for shortening and
filtering xtc files. It supports the 1995 and 2023 magic numbers.

> **NOTE:** _molly_ is in a pretty stable state and is used in the wild.
> Please do take care and verify the results. Blind trust in any tool is
> irresponsible.
>
> For any questions, feel free to get in [contact][contact] with me.

## Installation

### Command line application

```console
cargo install molly -F cli
```

#### Usage

With the _molly_ command, xtc files can be filtered and shortened. Selections
can be made on frames as well as the atoms within the frames.

- Frames can be selected with the `-f`/`--frame-selection` option, using
  `start:stop:step` ranges, which operate much like ranges in Python.
- The first _n_ atoms can be selected with the `-a`/`--atom-selection` option.

Here is a short showcase of possible uses.

```sh
# List all options.
molly --help

# Print a summary of a trajectory to standard out.
molly --info big.xtc

# Trajectories can be filtered in a number of ways. Here are a few combinations.
# Select the 100th to the 600th frame in steps of two. From those, store only the first 161 atoms.
molly big.xtc out.xtc --frame-selection 100:600:2 --atom-selection 161
molly big.xtc out.xtc -f 100:600:2 -a 161  # With shorter arguments.

# Reverse a selection. Here we use it to select the last frame.
molly big.xtc last.xtc --reverse-frame-selection --frame-selection :1
molly big.xtc last.xtc -Rf :1  # With shorter arguments.

# Reverse a trajectory.
molly big.xtc rev.xtc --reverse

# For any of these filtering commands, the frame times and steps can be written to standard out.
molly big.xtc rev_last_ten.xtc -rRf :10 --steps --times
```

### As a library

To use _molly_ in a Rust project, add this repository to the dependencies in
your `Cargo.toml`.
Find [_molly_ on crates.io][crates].

```rust
use molly::{XTCReader, XTCWriter, Frame};

// Read frames from a trajectory.
let mut reader = XTCReader::open("input.xtc")?;
let frames = reader.read_all_frames()?;

// Write frames to a new file.
let mut writer = XTCWriter::create("output.xtc")?;
for frame in frames.iter() {
    writer.write_frame(frame)?;
}
```

### As a Python module

`cargo` (which provides the Rust compiler) is required for building the Python
bindings. (The `stable` toolchain is sufficient.)

To install the module, run the following command. It will automatically clone
the repository and install the Python library from the correct directory.

```console
pip3 install 'git+https://git.sr.ht/~ma3ke/molly#egg=molly&subdirectory=bindings/python'
```

Alternatively, clone the repository, go into the bindings directory, and
install it from there using `pip`.

```console
git clone https://git.sr.ht/~ma3ke/molly
cd molly/bindings/python

# Perhaps you want to use/create a virtual environment first.
python3 -m venv venv
source venv/bin/activate

pip3 install .
```

### The examples

A number of useful example programs can be found in the `examples` directory.
Some of these can be used to benchmark against other xtc reader
implementations, or to create test files.

> **NOTE:** I'm leaving these here for the moment, but ultimately, I will
> remove or fundamentally change many of these examples.

In order to access these, clone the repository and build them.

```console
git clone https://git.sr.ht/~ma3ke/molly
cd molly
cargo build --release --examples
target/release/examples/<name> [options]
```

Or directly run them using

```console
cargo run --release --example <name>
```

## Tests

The library includes a number of unit tests of internal mechanisms and
integration tests (including comparisons against the values produced by other
libraries). Note that it is desirable to run the tests with the `--release`
flag, since the debug builds run rather slow.

```console
cargo test --release
```

## Performance and benchmarks

Go ahead and run the provided benchmarks if you're interested!

```console
cargo bench
```

Additionally, there is a couple of benchmark scripts lying around the repo. I
may place them into a neat table at a later point. For now, some things are
still subject to change. Though we can see the broad shape of the performance
story for _molly_, this is not yet the time for hard promises.

It looks like _molly_ is around 2&times; faster than [_xdrf_][xdrf]
(the widely-used Gromacs implementation), and around 4&times; faster than the
[_chemfiles_ implementation][chemfiles]. (The development dependency on
_chemfiles_ enabling benchmark comparisons was dropped after 0.6.0 due to its
instability.)

For the buffered implementations this gap is slightly less pronounced. When
disk I/O is factored out, buffered reading is around 20% slower than unbuffered
reading. But over very large trajectories where only a subset of positions from
the top of each frame is selected, the advantage is considerable.

## Thanks

- I want to thank [Ladislav Bartos](https://github.com/Ladme) for his
  contributions and feedback, including his fix of the large frame number–bug.
- Thanks to [Mikael Lund](https://github.com/mlund) for adding writing support.

[sourcehut]: https://git.sr.ht/~ma3ke/molly
[github]: https://github.com/ma3ke/molly
[contact]: https://dwangschematiek.nl/where
[xtc]: https://manual.gromacs.org/current/reference-manual/file-formats.html#xtc
[crates]: https://crates.io/crates/molly
[xdrf]: https://gitlab.com/gromacs/gromacs/-/blob/d8d6543db04563cb15f71c90ffb5ed2fda092bce/src/gromacs/fileio/xdrf.h
[chemfiles]: https://chemfiles.org/

---

Marieke Westendorp, 2024
