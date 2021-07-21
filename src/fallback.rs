use core::marker::PhantomData;
#[cfg(feature = "std")]
use std::sync::{Mutex, RwLock};

use array_macro::array;
use lazy_static::lazy_static;

/// Provides the necessary locking operations in case atomics aren't available
pub trait FallbackLock: Sync {
    /// Implementations should:
    /// * check if an optimistic read is possible
    /// * perform any necessary preparations to validate the optimistic read
    /// * call `f` (performing the read)
    /// * check that optimistic read was valid
    ///
    /// Implementations must return `Err(())` if the value is not validated.
    fn read_optimistic<R: Copy, F: FnOnce() -> R>(&self, _f: F) -> Result<R, ()> {
        Err(())
    }

    /// Implementations should gain the lock (ideally with shared read access if supported), then call `f`.
    fn read<R, F: FnOnce() -> R>(&self, f: F) -> R {
        self.write(f)
    }

    /// Implementations should gain the lock with exclusive access, then call `f`.
    fn write<R, F: FnOnce() -> R>(&self, f: F) -> R;
}

/// An abstraction over global lock sets
pub trait StaticLocks: 'static + FallbackLock {
    /// Return a lock based on the address chosen (helps share the load)
    fn get_lock(addr: usize) -> &'static Self;
}

#[cfg(feature = "std")]
impl<T: Sync + Send> FallbackLock for RwLock<T> {
    fn read<R, F: FnOnce() -> R>(&self, f: F) -> R {
        let _guard = self.read().unwrap();
        f()
    }
    fn write<R, F: FnOnce() -> R>(&self, f: F) -> R {
        let _guard = self.write().unwrap();
        f()
    }
}

#[cfg(feature = "std")]
impl<T: Send> FallbackLock for Mutex<T> {
    fn write<R, F: FnOnce() -> R>(&self, f: F) -> R {
        let _guard = self.lock().unwrap();
        f()
    }
}

macro_rules! static_lock_impl {
    ($lock:ident) => {
        impl StaticLocks for $lock<()> {
            fn get_lock(addr: usize) -> &'static Self {
                // The number of locks is a prime number because we want to make sure `addr % LEN` gets
                // dispersed across all locks.
                //
                // Note that addresses are always aligned to some power of 2, depending on type `T` in
                // `AtomicOptionCell<T>`. If `LEN` was an even number, then `addr % LEN` would be an even number,
                // too, which means only half of the locks would get utilized!
                //
                // It is also possible for addresses to accidentally get aligned to a number that is not a
                // power of 2. Consider this example:
                //
                // ```
                // #[repr(C)]
                // struct Foo {
                //     a: AtomicOptionCell<u8>,
                //     b: u8,
                //     c: u8,
                // }
                // ```
                //
                // Now, if we have a slice of type `&[Foo]`, it is possible that field `a` in all items gets
                // stored at addresses that are multiples of 3. It'd be too bad if `LEN` was divisible by 3.
                // In order to protect from such cases, we simply choose a large prime number for `LEN`.
                const LEN: usize = 97;

                lazy_static! {
                    static ref LOCKS: [$lock<()>; LEN] = array![_ => Default::default(); LEN];
                }

                // If the modulus is a constant number, the compiler will use crazy math to transform this into
                // a sequence of cheap arithmetic operations rather than using the slow modulo instruction.
                &LOCKS[addr % LEN]
            }
        }
    };
}

#[cfg(feature = "std")]
static_lock_impl! {Mutex}

#[cfg(feature = "std")]
static_lock_impl! {RwLock}

/// A ZST proxy to use the global lock set as a [`FallbackLock`]
/// [`FallbackLock`]: FallbackLock
#[derive(Copy, Clone, Default)]
pub struct GlobalFallbackLock<T>(PhantomData<T>);

impl<T> GlobalFallbackLock<T> {
    pub const fn new() -> GlobalFallbackLock<T> {
        GlobalFallbackLock(PhantomData)
    }
}

impl<T: StaticLocks> FallbackLock for GlobalFallbackLock<T> {
    fn read_optimistic<R: Copy, F: FnOnce() -> R>(&self, f: F) -> Result<R, ()> {
        let addr = self as *const _ as usize;
        T::get_lock(addr).read_optimistic(f)
    }
    fn read<R, F: FnOnce() -> R>(&self, f: F) -> R {
        let addr = self as *const _ as usize;
        T::get_lock(addr).read(f)
    }
    fn write<R, F: FnOnce() -> R>(&self, f: F) -> R {
        let addr = self as *const _ as usize;
        T::get_lock(addr).write(f)
    }
}
