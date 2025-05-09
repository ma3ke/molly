use std::collections::BTreeSet;
use std::num::NonZeroU64;

// Invariant: The selection is only valid if the frame it reads them into is appropriately sized.
// It is assumed that the frame is correctly sized, i.e.,
//     len(frame.atoms) == len(IndexList) == sum(Map) == Until
// but it is also fine if the frame is too large for the selection. Stated differently,
//     len(frame.atoms) >= len(IndexList)
//     len(frame.atoms) >= sum(Mask)
//     len(frame.atoms) >= Until
// However, for the section of the frame that is not accounted for by the selection, the output is
// undefined. This does not mean it is unsafe, but they cannot be interpreted as valid positions.
// For Map a further invariant exists:
//     len(Mask) <= len(encoded_atoms)
/// A selection of atoms.
#[derive(Debug, Default, Clone)]
pub enum AtomSelection {
    /// Include all atoms.
    #[default]
    All,
    /// A mask of the positions to include in the selection.
    ///
    /// If the value of the mask at an index `n` is `true`, the position at that same index `n` is
    /// included in the selection.
    Mask(Vec<bool>), // TODO: Bitmap optimization?
    /// Index of the position right after the last position to be included in the selection.
    ///
    /// This is an exclusive stop value, such that a value of 8 will mean that a total of 7 atoms
    /// are read into the frame.
    Until(u32),
}

impl AtomSelection {
    /// Create a boolean mask from a list of indices.
    pub fn from_index_list(indices: &[u32]) -> Self {
        let max = match indices.iter().max() {
            Some(&max) => max as usize + 1,
            None => return Self::Mask(Vec::new()),
        };
        let mut mask = Vec::with_capacity(max);
        mask.resize(max, false);

        for &idx in indices {
            mask[idx as usize] = true;
        }

        Self::Mask(mask)
    }

    /// Determine whether some index `idx` is included in this [`AtomSelection`].
    ///
    /// Will return [`None`] once the index is beyond the scope of this `AtomSelection`.
    pub fn is_included(&self, idx: usize) -> Option<bool> {
        match self {
            AtomSelection::All => Some(true),
            AtomSelection::Mask(mask) => mask.get(idx).copied(),
            AtomSelection::Until(until) => {
                if idx <= *until as usize {
                    Some(true)
                } else {
                    None
                }
            }
        }
    }

    /// Return the last index in this [`AtomSelection`].
    ///
    /// Note that this is not always equal to a `AtomSelection`'s `end` field. In some cases, the
    /// `step` of a `AtomSelection` may not neatly visit the index right before the `end`. This
    /// function will return the last index before the `end`, taking the value of `step` into
    /// account.
    pub fn last(&self) -> Option<usize> {
        match self {
            AtomSelection::All => None,
            AtomSelection::Mask(mask) => match mask.iter().rposition(|&entry| entry) {
                Some(n) => Some(n + 1),
                None => Some(0),
            },
            AtomSelection::Until(until) => Some(*until as usize),
        }
    }

    /// The number of positions selected by this [`AtomSelection`].
    ///
    /// This function will return at most `frame_natoms`.
    pub(crate) fn natoms_selected(&self, frame_natoms: usize) -> usize {
        match self {
            AtomSelection::All => frame_natoms,
            AtomSelection::Mask(mask) => mask
                .iter()
                .take(frame_natoms)
                .filter(|&&include| include)
                .count(),
            AtomSelection::Until(until) => usize::min(*until as usize, frame_natoms),
        }
    }

    /// The number of positions that must be read to fulfill this [`AtomSelection`].
    ///
    /// This function will return at most `frame_natoms`.
    ///
    /// Note that the return value for this function will only differ from
    /// [`AtomSelection::natoms_selected`] for the `AtomSelection::Mask` variant.
    pub(crate) fn reading_limit(&self, frame_natoms: usize) -> usize {
        // TODO: Verify that the natoms used here is well-conceived: it needs to be the number of
        // atoms that reside in the total compressed frame, but not the natoms we eventually want
        // to give back to the caller.
        self.last()
            .map(|n| usize::min(n, frame_natoms))
            .unwrap_or(frame_natoms)
    }
}

/// A selection of [`Frame`]s.
#[derive(Debug, Default, Clone)]
pub enum FrameSelection {
    /// Include all frames that are in a trajectory.
    #[default]
    All,
    /// Include frames that lie within a certain [`Range`].
    Range(Range),
    /// Include frames that match the indices in this list.
    ///
    /// Invariant: The indices in the FrameList are _unique_ and _consecutive_.
    FrameList(BTreeSet<usize>),
}

impl FrameSelection {
    /// Create a new `FrameSelection::FrameList` variant from an iterator.
    pub fn framelist_from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        Self::FrameList(BTreeSet::from_iter(iter))
    }

    /// Determine whether some index `idx` is included in this [`FrameSelection`].
    ///
    /// Will return [`None`] once the index is beyond the scope of this `FrameSelection`.
    pub fn is_included(&self, idx: usize) -> Option<bool> {
        match self {
            FrameSelection::All => Some(true),
            FrameSelection::Range(range) => range.is_included(idx as u64),
            FrameSelection::FrameList(indices) => {
                if *indices.last()? < idx {
                    None
                } else {
                    Some(indices.contains(&idx))
                }
            }
        }
    }

    /// If available, return the index of the frame up to which the frames should be read.
    ///
    /// This is an _exclusive_ value. If some index is returned, the index itself is not included
    /// in the [`FrameSelection`], but the frame before it is.
    pub fn until(&self) -> Option<usize> {
        match self {
            FrameSelection::All => None,
            FrameSelection::Range(range) => range.last().map(|last| last + 1),
            FrameSelection::FrameList(list) => {
                Some(list.iter().max().copied().unwrap_or_default() + 1)
            }
        }
    }
}

/// A selection of [`Frame`](super::Frame)s to be read from an [`XTCReader`](super::XTCReader).
///
/// The `start` of a [`Selection`] is always bounded, and is zero by default.
/// The `end` may be bounded or unbounded. In case the end is unbounded ([`None`]), a `Selection`
/// instructs the `XTCReader` to just read up to and including the last frame. If it is bounded
/// by [`Some`] value, the frames up to that index will be read.
/// The `step` describes the number of frames that passed in each stride.
/// The number of skipped `Frame`s is equal to `step` - 1.
/// For instance, given a `step` of four, one `Frame` is read and the following three are skipped.
///
/// # Note
///
/// An instance where `start` > `end` is a valid `Selection`, but it will not make much sense,
/// since the `Selection` will be understood to produce zero steps. This case will trigger a
/// `debug_assert`.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    /// The `start` of a [`Selection`] is always bounded, and is zero by default.
    pub start: u64,
    /// The `end` may be bounded or unbounded.
    ///
    /// In case the end is unbounded ([`None`]), a `Selection` instructs the `XTCReader` to just
    /// read up to and including the last frame. If it is bounded by [`Some`] value, the frames up
    /// to that index will be read. So, when `end` is bounded, it is an exclusive bound.
    pub end: Option<u64>,
    /// The `step` describes the number of frames that passed in each stride.
    ///
    /// The number of skipped `Frame`s is equal to `step` - 1.
    /// For instance, given a `step` of four, one `Frame` is read and the following three are skipped.
    pub step: NonZeroU64,
}

impl Range {
    pub fn new(start: Option<u64>, end: Option<u64>, step: Option<NonZeroU64>) -> Self {
        let mut sel = Self {
            end,
            ..Self::default()
        };
        if let Some(start) = start {
            sel.start = start;
        }
        if let Some(step) = step {
            sel.step = step;
        }

        if let Some(end) = sel.end {
            let start = sel.start;
            debug_assert!(
                start <= end,
                "the start of a selection ({start}) may not exceed the end ({end})"
            );
        }

        sel
    }

    /// If known, return whether some index is included in this range.
    pub fn is_included(&self, idx: u64) -> Option<bool> {
        if let Some(end) = self.end {
            // Determine whether `idx` is already beyond the defined range.
            if end <= idx {
                return None;
            }
        }
        let in_range = self.start <= idx;
        // Note that the subtraction of the start and idx should be fine in release mode, due to
        // the `in_range` precondition. The function must return `Some(false)` if that subtraction
        // would overflow. But we do use a saturating sub defensively, and to not panic on the
        // inconsequential overflow in debug builds.
        let in_step = self.step.get() == 1 || idx.saturating_sub(self.start) % self.step == 0;
        Some(in_range && in_step)
    }

    /// Return the last index in this [`Range`].
    ///
    /// Note that this is not always equal to a `Range`'s `end` field. In some cases, the `step` of
    /// a `Range` may not neatly visit the index right before the `end`. This function will return
    /// the last index before the `end`, taking the value of `step` into account.
    pub fn last(&self) -> Option<usize> {
        self.end.map(|end| {
            let length = end.saturating_sub(self.start);
            let remainder = length % self.step;
            (end - remainder) as usize
        })
    }
}

impl Default for Range {
    fn default() -> Self {
        Self {
            start: 0,
            end: None,
            step: NonZeroU64::new(1).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod frame {
        use std::num::NonZeroU64;

        use super::{FrameSelection, Range};

        #[test]
        fn zero_selection() {
            let list_empty = FrameSelection::FrameList(Default::default());
            let list_zero = FrameSelection::framelist_from_iter([0]);
            let range_empty = FrameSelection::Range(Range::new(None, Some(0), None));

            for idx in 0..1000 {
                assert!(list_empty.is_included(idx).is_none());
                if idx > 0 {
                    assert!(list_zero.is_included(idx).is_none());
                }
                assert!(range_empty.is_included(idx).is_none());
            }
        }

        #[test]
        fn first_n() {
            let n = 100;
            let step = NonZeroU64::new(17).unwrap();

            let list = FrameSelection::FrameList((0..n).collect());
            let until = FrameSelection::Range(Range::new(None, Some(n as u64), None));
            let from_n = FrameSelection::Range(Range::new(Some(n as u64), None, None));
            let until_stepped = FrameSelection::Range(Range::new(None, Some(n as u64), Some(step)));
            let from_n_stepped =
                FrameSelection::Range(Range::new(Some(n as u64), None, Some(step)));
            let all = FrameSelection::All;

            for idx in 0..2 * n {
                if idx < n {
                    assert_eq!(list.is_included(idx), Some(true));
                    assert_eq!(until.is_included(idx), Some(true));
                    assert_eq!(
                        until_stepped.is_included(idx),
                        Some(idx as u64 % step.get() == 0),
                    );
                } else {
                    assert!(list.is_included(idx).is_none());
                    assert!(until.is_included(idx).is_none());
                    assert!(until_stepped.is_included(idx).is_none());
                }
                let from_n_included = idx >= n;
                assert_eq!(from_n.is_included(idx), Some(from_n_included));
                assert_eq!(
                    from_n_stepped.is_included(idx),
                    Some(from_n_included && (idx as u64 - n as u64) % step.get() == 0),
                );
                assert_eq!(all.is_included(idx), Some(true));
            }
        }

        /// This test serves to replicate a degenerate case I found.
        #[test]
        fn range_clamped_step() {
            let end = 50;
            let s =
                FrameSelection::Range(Range::new(Some(25), Some(end), Some(3.try_into().unwrap())));
            assert_eq!(s.until(), Some(end as usize));

            let included = [25, 28, 31, 34, 37, 40, 43, 46, 49];
            for i in 0..60 {
                let expected = if i < end {
                    Some(included.contains(&i))
                } else {
                    None
                };
                assert_eq!(s.is_included(i as usize), expected);
            }
        }

        #[test]
        fn until() {
            let n = 100;
            let step = NonZeroU64::new(17).unwrap();

            let list = FrameSelection::FrameList((0..n).collect());
            let until = FrameSelection::Range(Range::new(None, Some(n as u64), None));
            let from_n = FrameSelection::Range(Range::new(Some(n as u64), None, None));
            let until_stepped = FrameSelection::Range(Range::new(None, Some(n as u64), Some(step)));
            let from_n_stepped =
                FrameSelection::Range(Range::new(Some(n as u64), None, Some(step)));
            let from_until_stepped =
                FrameSelection::Range(Range::new(Some(n as u64 / 3), Some(n as u64), Some(step)));
            let all = FrameSelection::All;

            assert_eq!(list.until(), Some(n));
            assert_eq!(until.until(), Some(n + 1));
            assert!(from_n.until().is_none());
            assert_eq!(until_stepped.until(), Some(86));
            assert!(from_n_stepped.until().is_none());
            assert_eq!(from_until_stepped.until(), Some(85));
            assert!(all.until().is_none());
        }
    }

    mod atom {
        use super::AtomSelection;

        #[test]
        fn zero_selection() {
            let m = 100;

            let mask_empty = AtomSelection::Mask(vec![]);
            let mask_false = AtomSelection::Mask(vec![false; m]);
            let list_empty = AtomSelection::from_index_list(&[]);
            let list_zero = AtomSelection::from_index_list(&[0]);
            let until_zero = AtomSelection::Until(0);

            for idx in 0..1000 {
                assert!(mask_empty.is_included(idx).is_none());
                if idx < m {
                    assert_eq!(mask_false.is_included(idx), Some(false));
                } else {
                    assert!(mask_false.is_included(idx).is_none());
                }
                assert!(list_empty.is_included(idx).is_none());
                if idx > 0 {
                    assert!(until_zero.is_included(idx).is_none());
                    assert!(list_zero.is_included(idx).is_none());
                } else {
                    assert_eq!(until_zero.is_included(idx), Some(true));
                    assert_eq!(list_zero.is_included(idx), Some(true));
                }
            }
        }

        #[test]
        fn first_n() {
            let n = 100;
            let mask = AtomSelection::Mask(vec![true; n]);
            let mask_trailing_false = AtomSelection::Mask([vec![true; n], vec![false; n]].concat());
            let list = AtomSelection::from_index_list(&(0..n as u32).collect::<Vec<_>>());
            let until = AtomSelection::Until(n as u32 - 1);
            let all = AtomSelection::All;

            for idx in 0..2 * n {
                if idx < n {
                    assert_eq!(mask.is_included(idx), Some(true));
                    assert_eq!(list.is_included(idx), Some(true));
                    assert_eq!(until.is_included(idx), Some(true));
                } else {
                    assert!(mask.is_included(idx).is_none());
                    assert!(list.is_included(idx).is_none());
                    assert!(until.is_included(idx).is_none());
                }
                assert_eq!(mask_trailing_false.is_included(idx), Some(idx < n));
                assert_eq!(all.is_included(idx), Some(true));
            }
        }

        #[test]
        fn non_continuous_mask() {
            let n = 100;

            let mask = AtomSelection::Mask(vec![
                true, true, true, false, false, false, true, false, false, true, false,
            ]);
            assert_eq!(mask.is_included(0), Some(true));
            assert_eq!(mask.is_included(1), Some(true));
            assert_eq!(mask.is_included(2), Some(true));
            assert_eq!(mask.is_included(3), Some(false));
            assert_eq!(mask.is_included(4), Some(false));
            assert_eq!(mask.is_included(5), Some(false));
            assert_eq!(mask.is_included(6), Some(true));
            assert_eq!(mask.is_included(7), Some(false));
            assert_eq!(mask.is_included(8), Some(false));
            assert_eq!(mask.is_included(9), Some(true));
            assert_eq!(mask.is_included(10), Some(false));
            assert_eq!(mask.is_included(11), None);
            assert_eq!(mask.is_included(12), None);
            assert_eq!(mask.is_included(100), None);
            let nselected = mask.natoms_selected(n);
            assert_eq!(nselected, 5);
            let limit = mask.reading_limit(n);
            assert_eq!(limit, 10);

            let s = 15; // In a 100, we can take 7 15-sized steps.
            let t = 7;
            let steps =
                AtomSelection::from_index_list(Vec::from_iter((0..n as u32).step_by(s)).as_slice());
            assert_eq!(steps.is_included(0), Some(true));
            assert_eq!(steps.is_included(1), Some(false));
            assert_eq!(steps.is_included(15), Some(true));
            assert_eq!(steps.is_included(16), Some(false));
            assert_eq!(steps.is_included(30), Some(true));
            assert_eq!(steps.is_included(32), Some(false));
            assert_eq!(steps.is_included(45), Some(true));
            assert_eq!(steps.is_included(48), Some(false));
            assert_eq!(steps.is_included(60), Some(true));
            assert_eq!(steps.is_included(70), Some(false));
            assert_eq!(steps.is_included(75), Some(true));
            assert_eq!(steps.is_included(80), Some(false));
            assert_eq!(steps.is_included(89), Some(false));
            assert_eq!(steps.is_included(90), Some(true));
            assert_eq!(steps.is_included(91), None);
            assert_eq!(steps.is_included(100), None);
            assert_eq!(steps.is_included(101), None);
            assert_eq!(steps.is_included(200), None);
            let nselected = steps.natoms_selected(n);
            assert_eq!(nselected, t);
            let limit = steps.reading_limit(n);
            assert_eq!(limit, 91);
        }
    }
}
