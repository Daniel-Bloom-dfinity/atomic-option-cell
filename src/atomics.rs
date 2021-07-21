use core::{
    mem, ptr,
    sync::atomic::{self, Ordering},
};

use crate::fallback::FallbackLock;

/// An atomic `()`.
///
/// All operations are noops.
struct AtomicUnit;

impl AtomicUnit {
    #[inline]
    fn load(&self, _order: Ordering) {}

    #[inline]
    fn store(&self, _val: (), _order: Ordering) {}

    #[inline]
    fn swap(&self, _val: (), _order: Ordering) {}

    #[inline]
    fn compare_exchange_weak(
        &self,
        _current: (),
        _new: (),
        _success: Ordering,
        _failure: Ordering,
    ) -> Result<(), ()> {
        Ok(())
    }

    #[inline]
    fn compare_exchange(
        &self,
        _current: (),
        _new: (),
        _success: Ordering,
        _failure: Ordering,
    ) -> Result<(), ()> {
        Ok(())
    }
}

macro_rules! atomic {
    // If values of type `$t` can be transmuted into values of the primitive atomic type `$atomic`,
    // declares variable `$a` of type `$atomic` and executes `$atomic_op`, breaking out of the loop.
    (@check, $t:ty, $atomic:ty, $a:ident, $atomic_op:expr) => {
        if can_transmute::<$t, $atomic>() {
            let $a: &$atomic;
            break $atomic_op;
        }
    };

    // If values of type `$t` can be transmuted into values of a primitive atomic type, declares
    // variable `$a` of that type and executes `$atomic_op`. Otherwise, just executes
    // `$fallback_op`.
    ($t:ty, $a:ident, $atomic_op:expr, $fallback_op:expr) => {
        loop {
            atomic!(@check, $t, AtomicUnit, $a, $atomic_op);
            atomic!(@check, $t, atomic::AtomicUsize, $a, $atomic_op);

            atomic!(@check, $t, atomic::AtomicU8, $a, $atomic_op);
            atomic!(@check, $t, atomic::AtomicU16, $a, $atomic_op);
            atomic!(@check, $t, atomic::AtomicU32, $a, $atomic_op);
            atomic!(@check, $t, atomic::AtomicU64, $a, $atomic_op);
            // TODO: AtomicU128 is unstable
            // atomic!(@check, $t, atomic::AtomicU128, $a, $atomic_op);

            break $fallback_op;
        }
    };
}

/// Returns `true` if values of type `A` can be transmuted into values of type `B`.
const fn can_transmute<A, B>() -> bool {
    // Sizes must be equal, but alignment of `A` must be greater or equal than that of `B`.
    (mem::size_of::<A>() == mem::size_of::<B>()) & (mem::align_of::<A>() >= mem::align_of::<B>())
}

/// Returns `true` if operations on `AtomicOptionCell<T>` are lock-free.
pub const fn atomic_option_is_lock_free<T>() -> bool {
    atomic! { Option<T>, _a, true, false }
}

/// Atomically reads data from `src`.
///
/// This operation uses the `Acquire` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
pub unsafe fn atomic_load<T: Copy, L: FallbackLock>(lock: &L, src: *mut Option<T>) -> Option<T> {
    atomic! {
        Option<T>, a,
        {
            a = &*(src as *const _ as *const _);
            mem::transmute_copy(&a.load(Ordering::Acquire))
        },
        {
            // Try doing an optimistic read first.
            // We need a volatile read here because other threads might concurrently modify the
            // value. In theory, data races are *always* UB, even if we use volatile reads and
            // discard the data when a data race is detected. The proper solution would be to
            // do atomic reads and atomic writes, but we can't atomically read and write all
            // kinds of data since `AtomicU8` is not available on stable Rust yet.
            if let Ok(val) = lock.read_optimistic(|| ptr::read_volatile(src)) {
                val
            } else {
                // Grab a regular read lock so that writers don't starve this load.
                lock.read(|| ptr::read(src))
            }
        }
    }
}

/// Atomically writes `val` to `dst`.
///
/// This operation uses the `Release` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
pub unsafe fn atomic_store<T, L>(lock: &L, dst: *mut Option<T>, val: Option<T>)
where
    L: FallbackLock,
{
    atomic! {
        Option<T>, a,
        {
            a = &*(dst as *const _ as *const _);
            a.store(mem::transmute_copy(&val), Ordering::Release);
            mem::forget(val);
        },
        lock.write(|| ptr::write(dst, val))
    }
}

/// Atomically swaps data at `dst` with `val`.
///
/// This operation uses the `AcqRel` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
pub unsafe fn atomic_swap<T, L>(lock: &L, dst: *mut Option<T>, val: Option<T>) -> Option<T>
where
    L: FallbackLock,
{
    atomic! {
        Option<T>, a,
        {
            a = &*(dst as *const _ as *const _);
            let res = mem::transmute_copy(&a.swap(mem::transmute_copy(&val), Ordering::AcqRel));
            mem::forget(val);
            res
        },
        lock.write(|| ptr::replace(dst, val))
    }
}

/// Atomically compares data at `dst` to `None` and, if equal byte-for-byte, exchanges data at
/// `dst` with `new`.
///
/// Returns `None` on success, or `Some(val)` on failure.
///
/// This operation uses the `AcqRel` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
pub unsafe fn atomic_compare_store_weak<T, L>(
    lock: &L,
    dst: *mut Option<T>,
    val: T,
) -> Option<Result<T, T>>
where
    L: FallbackLock,
{
    let val = Some(val);
    atomic! {
        Option<T>, a,
        {
            a = &*(dst as *const _ as *const _);
            match a.compare_exchange_weak(
                mem::transmute_copy(&Option::<T>::None),
                mem::transmute_copy(&val),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    mem::forget(val);
                    None
                },
                Err(e) => if mem::transmute_copy::<_, Option<T>>(&e).is_some() {
                    val.map(Ok)
                } else {
                    val.map(Err)
                },
            }
        },
        lock.write(|| if Option::is_none(&*dst) {
            ptr::write(dst, val);
            None
        } else {
            val.map(Ok)
        })
    }
}
/// Atomically compares data at `dst` to `None` and, if equal byte-for-byte, exchanges data at
/// `dst` with `new`.
///
/// Returns `None` on success, or `Some(val)` on failure.
///
/// This operation uses the `AcqRel` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
pub unsafe fn atomic_compare_store<T, L>(lock: &L, dst: *mut Option<T>, val: T) -> Option<T>
where
    L: FallbackLock,
{
    let val = Some(val);
    atomic! {
        Option<T>, a,
        {
            a = &*(dst as *const _ as *const _);
            match a.compare_exchange(
                mem::transmute_copy(&Option::<T>::None),
                mem::transmute_copy(&val),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    mem::forget(val);
                    None
                },
                Err(_) => val,
            }
        },
        lock.write(|| if Option::is_none(&*dst) {
            ptr::write(dst, val);
            None
        } else {
            val
        })
    }
}
