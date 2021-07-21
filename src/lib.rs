use core::{cell::UnsafeCell, mem};
#[cfg(feature = "std")]
use std::{
    panic::{RefUnwindSafe, UnwindSafe},
    sync,
};

mod atomics;
mod fallback;

/// A convenient type-alias
#[cfg(feature = "std")]
pub type Mutex = sync::Mutex<()>;

/// A convenient type-alias
#[cfg(feature = "std")]
pub type RwLock = sync::RwLock<()>;

pub use atomics::atomic_option_is_lock_free;
use atomics::{
    atomic_compare_store, atomic_compare_store_weak, atomic_load, atomic_store, atomic_swap,
};
pub use fallback::{FallbackLock, GlobalFallbackLock, StaticLocks};

/// A thread-safe mutable memory location for .
///
/// This type is similar to [`Cell<Option<T>>`], with the addition that it can also be shared among multiple threads.
///
/// Operations on `AtomicOptionCell`s use atomic instructions whenever possible, but will fall back to locks otherwise.
/// You can call [`AtomicOptionCell::<T>::is_lock_free()`] to check whether atomic instructions or locks will be used.
/// This is most useful for non-zero types like [`Box`], [`Rc`], and [`Arc`]
///
/// Atomic loads use the [`Acquire`] ordering and atomic stores use the [`Release`] ordering.
///
/// [`Cell<Option<T>>`]: std::cell::Cell
/// [`AtomicOptionCell::<T>::is_lock_free()`]: AtomicOptionCell::is_lock_free
/// [`Acquire`]: std::sync::atomic::Ordering::Acquire
/// [`Release`]: std::sync::atomic::Ordering::Release
/// [`Box`]: std::boxed::Box
/// [`Rc`]: std::rc::Rc
/// [`Arc`]: std::sync::Arc
#[repr(transparent)]
pub struct AtomicOptionCell<T, L> {
    /// The inner value.
    ///
    /// If the value in the `UnsafeCell` can be transmuted into a primitive atomic type, it will be treated as such.
    /// Otherwise, all potentially concurrent operations on that data will be protected by the fallback lock `L`.
    value: (UnsafeCell<Option<T>>, L),
}

unsafe impl<T: Send, L: FallbackLock> Send for AtomicOptionCell<T, L> {}
unsafe impl<T: Send, L: FallbackLock> Sync for AtomicOptionCell<T, L> {}

#[cfg(feature = "std")]
impl<T, L: FallbackLock> UnwindSafe for AtomicOptionCell<T, L> {}
#[cfg(feature = "std")]
impl<T, L: FallbackLock> RefUnwindSafe for AtomicOptionCell<T, L> {}

impl<T, L: FallbackLock + Default> Default for AtomicOptionCell<T, L> {
    fn default() -> Self {
        AtomicOptionCell::new(None, Default::default())
    }
}

impl<T, L> AtomicOptionCell<T, L> {
    /// Creates a new atomic cell initialized with `val`.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(7), Mutex::new(()));
    /// ```
    pub const fn new(val: Option<T>, l: L) -> AtomicOptionCell<T, L> {
        AtomicOptionCell {
            value: (UnsafeCell::new(val), l),
        }
    }

    /// Returns `true` if operations on values of this type are lock-free.
    ///
    /// If the compiler or the platform doesn't support the necessary atomic instructions,
    /// `AtomicOptionCell<T>` will use the fallback lock `L` for every potentially concurrent atomic operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::{NonZeroIsize, NonZeroUsize};
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// // This type is internally represented as `AtomicUsize` so we can just use atomic
    /// // operations provided by it.
    /// assert_eq!(AtomicOptionCell::<NonZeroUsize, Mutex>::is_lock_free(), true);
    ///
    /// // A wrapper struct around `isize`.
    /// struct Foo {
    ///     bar: NonZeroIsize,
    /// }
    /// // `AtomicOptionCell<Foo>` will be internally represented as `AtomicIsize`.
    /// assert_eq!(AtomicOptionCell::<Foo, Mutex>::is_lock_free(), true);
    ///
    /// // Operations on zero-sized types are always lock-free.
    /// assert_eq!(AtomicOptionCell::<(), Mutex>::is_lock_free(), true);
    ///
    /// // Option<T> messes up the alignment/size for most normal types, so atomic operations
    /// // on them will have to use fallback locks for synchronization.
    /// assert_eq!(AtomicOptionCell::<u8, Mutex>::is_lock_free(), false);
    /// assert_eq!(AtomicOptionCell::<u64, Mutex>::is_lock_free(), false);
    ///
    /// // Very large types cannot be represented as any of the standard atomic types, so atomic
    /// // operations on them will have to use fallback locks for synchronization.
    /// assert_eq!(AtomicOptionCell::<[u8; 1000], Mutex>::is_lock_free(), false);
    /// ```
    pub const fn is_lock_free() -> bool {
        atomic_option_is_lock_free::<T>()
    }
}

impl<T: Copy, L: FallbackLock> AtomicOptionCell<T, L> {
    /// Consumes the atomic and returns the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(7), Mutex::new(()));
    ///
    /// assert_eq!(a.load(), Some(7));
    /// ```
    pub fn load(&self) -> Option<T> {
        unsafe { atomic_load(&self.value.1, self.value.0.get()) }
    }
}

impl<T, L: FallbackLock> AtomicOptionCell<T, L> {
    /// Consumes the atomic and returns the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(7), Mutex::new(()));
    /// let v = a.into_inner();
    ///
    /// assert_eq!(v, Some(7));
    /// ```
    pub fn into_inner(self) -> Option<T> {
        self.value.0.into_inner()
    }

    /// Stores `val` into the atomic cell.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(7), Mutex::new(()));
    ///
    /// assert_eq!(a.take(), Some(7));
    /// a.store(8);
    /// assert_eq!(a.load(), Some(8));
    /// ```
    pub fn store<V: Into<Option<T>>>(&self, val: V) {
        let val = val.into();
        if mem::needs_drop::<T>() {
            drop(self.swap(val));
        } else {
            unsafe {
                atomic_store(&self.value.1, self.value.0.get(), val);
            }
        }
    }

    /// If the cell is empty, stores `val` into the atomic cell.
    ///
    /// Returns `None` if `val` was stored successfully.
    /// Returns `Some(Ok(val))` in the case of failure to store because the cell already contains a value.
    /// Returns `Some(Err(val))` in the case of failure to store spuriously.
    /// `try_store_weak` is allowed to fail spuriously even when the cell is empty, which allows the compiler to generate better assembly code when the try store is used in a loop.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(None, Mutex::new(()));
    ///
    /// assert_eq!(a.load(), None);
    /// let mut v = 8;
    /// while let Some(Err(t)) = a.try_store_weak(v) {
    ///     v = t;
    /// }
    /// assert_eq!(a.load(), Some(8));
    /// assert_eq!(a.try_store_weak(7), Some(Ok(7)));
    /// ```
    pub fn try_store_weak(&self, val: T) -> Option<Result<T, T>> {
        unsafe { atomic_compare_store_weak(&self.value.1, self.value.0.get(), val) }
    }

    /// If the cell is empty, stores `val` into the atomic cell.
    ///
    /// Returns `None` if `val` was stored successfully.
    /// Returns `Some(val)` in the case of failure to store because the cell already contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(None, Mutex::new(()));
    ///
    /// assert_eq!(a.try_store(8), None);
    /// assert_eq!(a.try_store(7), Some(7));
    /// ```
    pub fn try_store(&self, val: T) -> Option<T> {
        unsafe { atomic_compare_store(&self.value.1, self.value.0.get(), val) }
    }

    /// Stores `val` into the atomic cell and returns the previous value.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(7), Mutex::new(()));
    ///
    /// assert_eq!(a.load(), Some(7));
    /// assert_eq!(a.swap(8), Some(7));
    /// assert_eq!(a.load(), Some(8));
    /// ```
    pub fn swap<V: Into<Option<T>>>(&self, val: V) -> Option<T> {
        unsafe { atomic_swap(&self.value.1, self.value.0.get(), val.into()) }
    }

    /// Takes the value of the atomic cell, leaving `Default::default()` in its place.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_option_cell::{AtomicOptionCell, Mutex};
    ///
    /// let a = AtomicOptionCell::new(Some(5), Mutex::new(()));
    /// let five = a.take();
    ///
    /// assert_eq!(five, Some(5));
    /// assert_eq!(a.into_inner(), None);
    /// ```
    pub fn take(&self) -> Option<T> {
        self.swap(None)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        mem,
        num::{
            NonZeroI16, NonZeroI32, NonZeroI8, NonZeroIsize, NonZeroU16, NonZeroU32, NonZeroU64,
            NonZeroU8, NonZeroUsize,
        },
        sync::atomic::{AtomicUsize, Ordering::SeqCst},
    };

    use super::*;

    #[test]
    fn is_lock_free() {
        struct UsizeWrap(NonZeroUsize);
        struct U8Wrap(bool);
        struct I16Wrap(NonZeroI16);
        #[repr(align(8))]
        struct U64Align8(NonZeroU64);

        assert!(atomic_option_is_lock_free::<NonZeroUsize>());
        assert!(atomic_option_is_lock_free::<NonZeroIsize>());
        assert!(atomic_option_is_lock_free::<UsizeWrap>());

        assert!(atomic_option_is_lock_free::<()>());

        assert!(atomic_option_is_lock_free::<NonZeroU8>());
        assert!(atomic_option_is_lock_free::<NonZeroI8>());
        assert!(atomic_option_is_lock_free::<bool>());
        assert!(atomic_option_is_lock_free::<U8Wrap>());

        assert!(atomic_option_is_lock_free::<NonZeroU16>());
        assert!(atomic_option_is_lock_free::<NonZeroI16>());
        assert!(atomic_option_is_lock_free::<I16Wrap>());

        assert!(atomic_option_is_lock_free::<NonZeroU32>());
        assert!(atomic_option_is_lock_free::<NonZeroI32>());

        // Sizes of both types must be equal, and the alignment of `u64` must be greater or equal than
        // that of `AtomicU64`. In i686-unknown-linux-gnu, the alignment of `u64` is `4` and alignment
        // of `AtomicU64` is `8`, so `AtomicOptionCell<u64>` is not lock-free.
        assert_eq!(
            atomic_option_is_lock_free::<NonZeroU64>(),
            cfg!(any(
                target_pointer_width = "64",
                target_pointer_width = "128"
            ))
        );
        assert_eq!(mem::size_of::<U64Align8>(), 8);
        assert_eq!(mem::align_of::<U64Align8>(), 8);
        assert!(atomic_option_is_lock_free::<U64Align8>());

        // AtomicU128 is unstable
        assert!(!atomic_option_is_lock_free::<u128>());
    }

    #[test]
    fn const_is_lock_free() {
        const _U: bool = atomic_option_is_lock_free::<NonZeroUsize>();
        const _I: bool = atomic_option_is_lock_free::<NonZeroIsize>();
    }

    #[test]
    fn drops_unit() {
        static CNT: AtomicUsize = AtomicUsize::new(0);
        CNT.store(0, SeqCst);

        #[derive(Debug, PartialEq, Eq)]
        struct Foo();

        impl Foo {
            fn new() -> Foo {
                CNT.fetch_add(1, SeqCst);
                Foo()
            }
        }

        impl Drop for Foo {
            fn drop(&mut self) {
                CNT.fetch_sub(1, SeqCst);
            }
        }

        impl Default for Foo {
            fn default() -> Foo {
                Foo::new()
            }
        }

        let a = AtomicOptionCell::new(Some(Foo::new()), Mutex::new(()));

        assert_eq!(a.swap(Foo::new()), Some(Foo::new()));
        assert_eq!(CNT.load(SeqCst), 1);

        a.store(Foo::new());
        assert_eq!(CNT.load(SeqCst), 1);

        assert_eq!(a.swap(Foo::default()), Some(Foo::new()));
        assert_eq!(CNT.load(SeqCst), 1);

        drop(a);
        assert_eq!(CNT.load(SeqCst), 0);
    }

    #[test]
    fn drops_u8() {
        static CNT: AtomicUsize = AtomicUsize::new(0);
        CNT.store(0, SeqCst);

        #[derive(Debug, PartialEq, Eq)]
        struct Foo(u8);

        impl Foo {
            fn new(val: u8) -> Foo {
                CNT.fetch_add(1, SeqCst);
                Foo(val)
            }
        }

        impl Drop for Foo {
            fn drop(&mut self) {
                CNT.fetch_sub(1, SeqCst);
            }
        }

        impl Default for Foo {
            fn default() -> Foo {
                Foo::new(0)
            }
        }

        let a = AtomicOptionCell::new(Some(Foo::new(5)), Mutex::new(()));

        assert_eq!(a.swap(Foo::new(6)), Some(Foo::new(5)));
        assert_eq!(a.swap(Foo::new(1)), Some(Foo::new(6)));
        assert_eq!(CNT.load(SeqCst), 1);

        a.store(Foo::new(2));
        assert_eq!(CNT.load(SeqCst), 1);

        assert_eq!(a.swap(Foo::default()), Some(Foo::new(2)));
        assert_eq!(CNT.load(SeqCst), 1);

        assert_eq!(a.swap(Foo::default()), Some(Foo::new(0)));
        assert_eq!(CNT.load(SeqCst), 1);

        drop(a);
        assert_eq!(CNT.load(SeqCst), 0);
    }

    #[test]
    fn drops_usize() {
        static CNT: AtomicUsize = AtomicUsize::new(0);
        CNT.store(0, SeqCst);

        #[derive(Debug, PartialEq, Eq)]
        struct Foo(usize);

        impl Foo {
            fn new(val: usize) -> Foo {
                CNT.fetch_add(1, SeqCst);
                Foo(val)
            }
        }

        impl Drop for Foo {
            fn drop(&mut self) {
                CNT.fetch_sub(1, SeqCst);
            }
        }

        impl Default for Foo {
            fn default() -> Foo {
                Foo::new(0)
            }
        }

        let a = AtomicOptionCell::new(Some(Foo::new(5)), Mutex::new(()));

        assert_eq!(a.swap(Foo::new(6)), Some(Foo::new(5)));
        assert_eq!(a.swap(Foo::new(1)), Some(Foo::new(6)));
        assert_eq!(CNT.load(SeqCst), 1);

        a.store(Foo::new(2));
        assert_eq!(CNT.load(SeqCst), 1);

        assert_eq!(a.swap(Foo::default()), Some(Foo::new(2)));
        assert_eq!(CNT.load(SeqCst), 1);

        assert_eq!(a.swap(Foo::default()), Some(Foo::new(0)));
        assert_eq!(CNT.load(SeqCst), 1);

        drop(a);
        assert_eq!(CNT.load(SeqCst), 0);
    }

    #[test]
    fn garbage_padding() {
        #[derive(Copy, Clone, Eq, PartialEq)]
        struct Object {
            a: i64,
            b: i32,
        }

        let cell = AtomicOptionCell::new(Some(Object { a: 0, b: 0 }), Mutex::new(()));
        let _garbage = [0xfe, 0xfe, 0xfe, 0xfe, 0xfe]; // Needed
        let next = Object { a: 0, b: 0 };

        cell.take();
        assert!(cell.try_store(next).is_none());
    }

    #[test]
    fn const_atomic_cell_new() {
        static CELL: AtomicOptionCell<usize, GlobalFallbackLock<Mutex>> =
            AtomicOptionCell::new(Some(0), GlobalFallbackLock::new());

        CELL.store(1);
        assert_eq!(CELL.load(), Some(1));
    }
}
