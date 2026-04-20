use std::sync::Arc;

use atomic_float::AtomicF64;

pub mod dgd;
pub mod extra;

pub trait OptAlg {
    fn step(&mut self);
    fn res(&self, res: &Arc<[AtomicF64]>);
}
