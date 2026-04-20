use std::sync::Arc;

use atomic_float::AtomicF32;

pub mod dgd;

pub trait OptAlg {
    fn step(&mut self);
    fn res(&self, res: &Arc<[AtomicF32]>);
}
