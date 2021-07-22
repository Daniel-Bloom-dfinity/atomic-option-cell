# Atomic Option Cell

Forked from [crossbeam-utils](https://github.com/crossbeam-rs/crossbeam)'s [`AtomicCell`](https://docs.rs/crossbeam/0.8.1/crossbeam/atomic/struct.AtomicCell.html), this crate provides a tweaked version of that structure called `AtomicOptionCell`, which has been optimized for atomically-sized `Option<NonZero>` types.

Where possible, this crate uses lockless operations, and exposes `AtomicOptionCell::is_lock_free` and `atomic_option_is_lock_free` to validate when they will be used.

In the interest of interoperability, the `FallbackLock` trait enables customized fallback locking solutions for non-atomically-sized types.

## Example

### Create a cell with a per-object `Mutex` as the fallback.

```rust
use atomic_option_cell::{AtomicOptionCell, Mutex};

let a = AtomicOptionCell::new(Some(5), Mutex::new(()));
let five = a.take();

assert_eq!(five, Some(5));
assert_eq!(a.into_inner(), None);
```

### Create a cell with a global `Mutex` set as the fallback.

```rust
use atomic_option_cell::{AtomicOptionCell, Mutex};

let a = AtomicOptionCell::new(Some(5), Mutex::new(()));
let five = a.take();

assert_eq!(five, Some(5));
assert_eq!(a.into_inner(), None);
```
