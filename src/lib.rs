//! A thread-safe indexed object pool with automatic return and attach/detach semantics.
//!
//! The goal of an object pool is to reuse expensive to allocate objects or frequently allocated objects.
//! This pool implementation allows to index different object pools of the same type as subpools.
//! For example you could use this to pool SSH connection objects, indexed by ipaddress:port.
//!
//! This pool is bound to sizes to avoid to keep allocations in control. There are 2 different
//! sizes that can be setup at creation time, `max_pool_indexes` and `max_single_pool_size`, the
//! former controlling how many indexes the pool can contain and the latter controlling the max
//! size of the subpool of a single index.
//!
//! On top of that indexes have an expiration time expressed as a duration.
//!
//! # Examples
//!
//! ## Creating a Pool
//!
//! The default pool creation looks like this, with 32 indexes with a capacity of 8 elements
//! expiring after 5 minutes from creation :
//! ```no_run
//! use index_pool::Pool;
//!
//! let pool: Pool<String> = Pool::default();
//! ```
//! Example pool with 64 `Vec<u8>` with capacity of 64 elements each expiring after 10 seconds:
//! ```
//! use std::time::Duration;
//! use index_pool::Pool;
//!
//! let pool: Pool<u8> = Pool::new(64, 64, Duration::from_secs(10));
//! ```
//!
//! ## Using a Pool
//!
//! Basic usage for pulling from the pool
//! ```
//! use index_pool::Pool;
//!
//! struct Obj {
//!     size: usize,
//! }
//!
//! let pool: Pool<Obj> = Pool::default();
//! let obj = pool.pull("item1", || Obj{size: 1});
//! assert_eq!(obj.size, 1);
//! let obj2 = pool.pull("item1", || Obj{size: 1});
//! assert_eq!(obj.size, 1);
//! ```
//! Pull from pool and `detach()`
//! ```
//! use index_pool::Pool;
//!
//! struct Obj {
//!     size: usize,
//! }
//!
//! let pool: Pool<Obj> = Pool::default();
//! let obj = pool.pull("item1", || Obj{size: 1});
//! assert_eq!(obj.size, 1);
//! let obj2 = pool.pull("item1", || Obj{size: 1});
//! assert_eq!(obj.size, 1);
//! let (pool, obj) = obj.detach();
//! assert_eq!(obj.size, 1);
//! pool.attach("item1", obj);
//! ```
//!
//! ## Using Across Threads
//!
//! You simply wrap the pool in a [`std::sync::Arc`]
//! ```no_run
//! use std::sync::Arc;
//! use index_pool::Pool;
//!
//! let pool: Arc<Pool<String>> = Arc::new(Pool::default());
//! ```
//!
//! # Warning
//!
//! Objects in the pool are not automatically reset, they are returned but NOT reset
//! You may want to call `object.reset()` or  `object.clear()`
//! or any other equivalent for the object that you are using, after pulling from the pool
//!
//! [`std::sync::Arc`]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html
extern crate log;
extern crate parking_lot;

use std::collections::BTreeMap;
use std::mem::{forget, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

#[cfg(not(test))]
use log::{debug, error, info};

#[cfg(test)]
use std::{println as info, println as error, println as debug};

const DEFAULT_POOL_INDEXES: usize = 32;
const DEFAULT_SINGLE_POOL_SIZE: usize = 8;
const DEFAULT_EXPIRATION: Duration = Duration::from_secs(300);

#[derive(Debug)]
struct Inner<T> {
    inner: T,
    start_time: Instant,
}

impl<T> Inner<T> {
    pub fn new(object: T, start_time: Instant) -> Self {
        Self { inner: object, start_time }
    }

    pub fn expired(&self, duration: Duration) -> bool {
        self.start_time.elapsed().as_millis() > duration.as_millis()
    }
}

#[derive(Debug)]
pub struct Pool<T> {
    max_pool_indexes: usize,
    max_single_pool_size: usize,
    expiration: Duration,
    objects: RwLock<BTreeMap<String, Vec<Inner<T>>>>,
}

impl<T> Pool<T> {
    pub fn new(max_pool_indexes: usize, max_single_pool_size: usize, expiration: Duration) -> Pool<T> {
        Pool { max_pool_indexes, max_single_pool_size, expiration, objects: RwLock::new(BTreeMap::new()) }
    }

    pub fn default() -> Pool<T> {
        Pool::new(DEFAULT_POOL_INDEXES, DEFAULT_SINGLE_POOL_SIZE, DEFAULT_EXPIRATION)
    }

    pub fn size(&self) -> usize {
        self.objects.read().len()
    }

    pub fn len(&self, item: &str) -> usize {
        match self.objects.read().get(item) {
            Some(item) => item.len(),
            None => 0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.size() >= self.max_pool_indexes
    }

    fn expunge_oldest(&self) {
        if !self.is_full() {
            debug!("Object pool is not full, nothing to remove");
            return;
        }
        let mut last = String::new();
        if let Some((obj, _)) = self.objects.read().iter().next() {
            last = obj.clone();
        }
        if !last.is_empty() {
            debug!("Removing oldest element in the queue: {}", last);
            self.objects.write().remove(&last);
        } else {
            error!("Unable to find an element to remove from the queue, next allocation could fail");
        }
    }

    fn try_pull(&self, item: &str) -> Option<Reusable<T>> {
        match self.objects.write().get_mut(item) {
            Some(objects) => {
                info!("Pool for {} is currently of {} objects", item, objects.len());
                if objects.len() > self.max_single_pool_size {
                    objects.pop();
                }
                match objects.pop() {
                    Some(object) => {
                        if object.expired(self.expiration) {
                            info!(
                                "Element {} has reached expiration time of {} ms, evicting from pool",
                                item,
                                object.start_time.elapsed().as_millis()
                            );
                            None
                        } else {
                            info!(
                                "Reusing element pool {} created {} ms ago",
                                item,
                                object.start_time.elapsed().as_millis()
                            );
                            Some(Reusable::new(self, item.to_string(), object.start_time, object.inner))
                        }
                    }
                    None => {
                        debug!("Element {} pool is empty", item);
                        None
                    }
                }
            }
            None => {
                debug!("Unable to find element {} in objects pool", item);
                None
            }
        }
    }

    fn attach_time(&self, item: &str, start_time: Instant, t: T) {
        self.expunge_oldest();
        if self.objects.read().contains_key(item) {
            debug!("Creating new pool of {} elements and attatching object to it", self.max_single_pool_size);
            self.objects.write().get_mut(item).unwrap().push(Inner::new(t, start_time))
        } else {
            debug!(
                "Attatching element {} to existing pool of max {} elements",
                self.len(item) + 1,
                self.max_single_pool_size
            );
            self.objects.write().insert(item.to_string(), vec![Inner::new(t, start_time)]);
        }
    }

    pub fn pull<F: Fn() -> T>(&self, item: &str, fallback: F) -> Reusable<T> {
        match self.try_pull(item) {
            Some(object) => object,
            None => {
                info!("Creating new element {} with a pool of {} instances", item, self.max_single_pool_size);
                for _ in 0..self.max_single_pool_size {
                    self.attach(item, fallback())
                }
                self.pull(item, fallback)
            }
        }
    }

    pub fn attach(&self, item: &str, t: T) {
        self.attach_time(item, Instant::now(), t);
    }
}

pub struct Reusable<'a, T> {
    item: String,
    pool: &'a Pool<T>,
    data: ManuallyDrop<T>,
    start_time: Instant,
}

impl<'a, T> Reusable<'a, T> {
    pub fn new(pool: &'a Pool<T>, item: String, start_time: Instant, t: T) -> Self {
        Self { item, pool, data: ManuallyDrop::new(t), start_time }
    }

    pub fn detach(mut self) -> (&'a Pool<T>, T) {
        let ret = unsafe { (self.pool, self.take()) };
        info!("Detaching object from element {} pool", self.item);
        forget(self);
        ret
    }

    unsafe fn take(&mut self) -> T {
        ManuallyDrop::take(&mut self.data)
    }
}

impl<'a, T> Deref for Reusable<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T> DerefMut for Reusable<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T> Drop for Reusable<'a, T> {
    fn drop(&mut self) {
        let value = unsafe { self.take() };
        info!("Re-attatching object to element {} object pool", self.item);
        self.pool.attach_time(&self.item, self.start_time, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::drop;
    use std::thread;

    use pretty_assertions::assert_eq;

    #[derive(Debug)]
    struct Obj {
        idx: usize,
    }

    impl Obj {
        fn new(idx: usize) -> Self {
            Self { idx }
        }
    }

    #[test]
    fn test_detach() {
        let pool = Pool::default();
        let (pool, object) = pool.pull("item1", || Obj::new(1)).detach();
        assert_eq!(object.idx, 1);
        assert_eq!(pool.len("item1"), 7);
        drop(object);
        assert_eq!(pool.len("item1"), 7);
    }

    #[test]
    fn test_detach_then_attach() {
        let pool = Pool::default();
        let (pool, object) = pool.pull("item1", || Obj::new(1)).detach();
        assert_eq!(object.idx, 1);
        assert_eq!(pool.len("item1"), 7);
        pool.attach("item1", object);
        assert_eq!(pool.try_pull("item1").unwrap().idx, 1);
        assert_eq!(pool.len("item1"), 8);
    }

    #[test]
    fn test_pull_and_size() {
        let pool = Pool::default();
        pool.attach("item1", Obj::new(1));
        assert_eq!(pool.size(), 1);

        let object1 = pool.try_pull("item1");
        let object2 = pool.try_pull("item1");
        let object3 = pool.pull("item2", || Obj::new(2));
        assert_eq!(pool.size(), 2);

        assert_eq!(object1.is_some(), true);
        assert_eq!(object2.is_none(), true);

        assert_eq!(pool.len("item1"), 0);
        drop(object1);
        assert_eq!(pool.len("item1"), 1);
        drop(object2);
        assert_eq!(pool.len("item1"), 1);

        assert_eq!(object3.idx, 2);
        assert_eq!(pool.len("item2"), 7);
        drop(object3);
        assert_eq!(pool.len("item2"), 8);
    }

    #[test]
    fn test_fill_up_pool() {
        let pool = Pool::default();
        for x in 0..DEFAULT_POOL_INDEXES {
            pool.attach(&format!("item{}", x), Obj::new(x));
            assert_eq!(pool.size(), x + 1)
        }
        for (_, obj) in pool.objects.read().iter() {
            assert_eq!(obj.len(), 1);
        }
    }

    #[test]
    fn test_expire_pool() {
        let pool = Pool::new(DEFAULT_POOL_INDEXES, DEFAULT_SINGLE_POOL_SIZE, Duration::from_secs(1));
        for x in 1..7 {
            pool.attach(&format!("item{}", x), Obj::new(x));
        }
        assert_eq!(pool.size(), 6);
        thread::sleep(Duration::from_millis(1500));
        for x in 1..7 {
            assert_eq!(pool.try_pull(&format!("item{})", x)).is_none(), true);
        }
        for x in 1..7 {
            pool.pull(&format!("item{})", x), || Obj::new(x));
        }
        for x in 1..7 {
            assert_eq!(pool.try_pull(&format!("item{})", x)).is_some(), true);
        }
    }

    #[test]
    fn test_smoke() {
        let pool = Pool::default();
        for x in 0..10000 {
            let obj = pool.pull(&format!("item{}", x), || Obj::new(x));
            assert_eq!(obj.data.idx, x);
            if x >= 32 {
                assert!(pool.size() >= 31);
            }
        }
    }
}
