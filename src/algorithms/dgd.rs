use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier, mpsc};

use atomic_float::AtomicF32;
use nalgebra as na;

use super::OptAlg;
use crate::graph::Graph;
use crate::objectives::ObjectiveFunction;

const EPS: f32 = 1e-5;

#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct DGDNode<O: ObjectiveFunction, G: Fn(usize, f32) -> f32> {
    id: usize,
    obj: O,
    x: na::DVector<f32>,
    x_true: na::DVector<f32>,
    w: na::DVector<f32>,
    k: usize,
    alpha_zero: f32,
    alpha_update: G,
    send: Box<[mpsc::Sender<(usize, na::DVector<f32>)>]>,
    recv: mpsc::Receiver<(usize, na::DVector<f32>)>,
    sync: Arc<Barrier>,
}

#[allow(clippy::type_complexity)]
impl<O: ObjectiveFunction, G: Fn(usize, f32) -> f32 + Copy> DGDNode<O, G> {
    pub fn new(
        graph: Graph,
        mut objs: Vec<O>,
        x_0: na::DVector<f32>,
        x_true: na::DVector<f32>,
        alpha_zero: f32,
        alpha_update: G,
    ) -> Box<[Self]> {
        let n = graph.adjacency_matrix.nrows();
        assert_eq!(objs.len(), n);

        let w = graph.weight_matrix();

        let mut senders: Vec<Vec<mpsc::Sender<(usize, na::DVector<f32>)>>> = vec![Vec::new(); n];
        let mut receivers: Vec<mpsc::Receiver<(usize, na::DVector<f32>)>> = Vec::with_capacity(n);

        for i in 0..n {
            let (sender, receiver) = mpsc::channel();
            receivers.push(receiver);
            for j in 0..n {
                if i == j {
                    continue;
                }
                if w[(i, j)].abs() < EPS {
                    continue;
                }
                senders[j].push(sender.clone());
            }
        }

        let sync = Arc::new(Barrier::new(n));
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let id = n - i - 1;
            let node = DGDNode {
                id,
                obj: objs.pop().expect("valid for all n"),
                x: x_0.clone(),
                x_true: x_true.clone(),
                w: w.column(id).clone_owned(),
                k: 0,
                alpha_zero,
                alpha_update,
                send: senders.pop().expect("valid for n").into_boxed_slice(),
                recv: receivers.pop().expect("valid for n"),
                sync: Arc::clone(&sync),
            };
            nodes.push(node);
        }

        nodes.into_boxed_slice()
    }
}

impl<O: ObjectiveFunction, G: Fn(usize, f32) -> f32> OptAlg for DGDNode<O, G> {
    fn step(&mut self) {
        self.k += 1;
        for s in self.send.iter() {
            s.send((self.id, self.x.clone())).expect("can send");
        }

        let mut new_x = self.w[self.id] * &self.x
            - (self.alpha_update)(self.k, self.alpha_zero) * self.obj.grad(&self.x);
        for _ in 0..self.send.len() {
            let (id, other_x) = self.recv.recv().expect("neighbors send");
            assert_ne!(id, self.id);
            new_x += self.w[id] * other_x;
        }
        self.x = new_x;
        self.sync.wait();
    }

    fn res(&self, res: &Arc<[AtomicF32]>) {
        res.get(self.k)
            .expect("do not exceed ITERATIONS + 1 length")
            .fetch_add((&self.x - &self.x_true).norm_squared(), Ordering::Relaxed);
    }
}
