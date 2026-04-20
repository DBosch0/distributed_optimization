use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier, mpsc};

use atomic_float::AtomicF64;
use nalgebra as na;

use super::OptAlg;
use crate::graph::Graph;
use crate::objectives::ObjectiveFunction;

const EPS: f64 = 1e-5;

pub struct ExtraNode<O: ObjectiveFunction> {
    id: usize,
    obj: O,
    x: na::DVector<f64>,
    // x_prev: Vec<na::DVector<f64>>,
    x_true: na::DVector<f64>,
    w: na::DVector<f64>,
    wtilde: na::DVector<f64>,
    k: usize,
    alpha: f64,
    grad_prev: na::DVector<f64>,
    wtilde_sum_prev: na::DVector<f64>,
    send: Box<[mpsc::Sender<(usize, na::DVector<f64>)>]>,
    recv: mpsc::Receiver<(usize, na::DVector<f64>)>,
    sync: Arc<Barrier>,
}

impl<O: ObjectiveFunction> ExtraNode<O> {
    pub fn new(
        graph: Graph,
        mut objs: Vec<O>,
        x_0: na::DVector<f64>,
        x_true: na::DVector<f64>,
        alpha: f64,
    ) -> Box<[Self]> {
        let n = graph.adjacency_matrix.nrows();
        let p = x_0.nrows();
        assert_eq!(objs.len(), n);

        let w = graph.weight_matrix();
        let wtilde = (&w + na::DMatrix::identity(n, n)) / 2.0;

        let mut senders: Vec<Vec<mpsc::Sender<(usize, na::DVector<f64>)>>> = vec![Vec::new(); n];
        let mut receivers: Vec<mpsc::Receiver<(usize, na::DVector<f64>)>> = Vec::with_capacity(n);

        for i in 0..n {
            let (sender, receiver) = mpsc::channel();
            receivers.push(receiver);
            for j in 0..n {
                if i == j {
                    continue;
                }
                if w[(i, j)].abs() < EPS {
                    assert!(wtilde[(i, j)].abs() < EPS);
                    continue;
                }
                senders[j].push(sender.clone());
            }
        }

        let sync = Arc::new(Barrier::new(n));
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let id = n - i - 1;
            let node = ExtraNode {
                id,
                obj: objs.pop().expect("valid for all n"),
                x: x_0.clone(),
                // x_prev: vec![na::DVector::zeros(p); n],
                x_true: x_true.clone(),
                w: w.column(id).clone_owned(),
                wtilde: wtilde.column(id).clone_owned(),
                k: 0,
                alpha,
                grad_prev: na::DVector::zeros(p),
                wtilde_sum_prev: na::DVector::zeros(p),
                send: senders.pop().expect("valid for n").into_boxed_slice(),
                recv: receivers.pop().expect("valid for n"),
                sync: Arc::clone(&sync),
            };
            nodes.push(node);
        }

        nodes.into_boxed_slice()
    }
}

impl<O: ObjectiveFunction> OptAlg for ExtraNode<O> {
    fn step(&mut self) {
        self.k += 1;

        for s in self.send.iter() {
            s.send((self.id, self.x.clone())).expect("can send");
        }

        let grad = self.obj.grad(&self.x);

        let (new_x, new_w_tilde_sum) = if self.k == 1 {
            let mut new_x = self.w[self.id] * &self.x - self.alpha * &grad;
            let mut new_w_tilde_sum = self.wtilde[self.id] * &self.x;
            for _ in 0..self.send.len() {
                let (id, other_x) = self.recv.recv().expect("neighbors send");
                new_x += self.w[id] * &other_x;
                new_w_tilde_sum += self.wtilde[id] * &other_x
            }
            (new_x, new_w_tilde_sum)
        } else {
            let mut new_x = (1.0 + self.w[self.id]) * &self.x
                - &self.wtilde_sum_prev
                - self.alpha * (&grad - &self.grad_prev);
            let mut new_w_tilde_sum = self.wtilde[self.id] * &self.x;
            for _ in 0..self.send.len() {
                let (id, other_x) = self.recv.recv().expect("neighbors send");
                new_x += self.w[id] * &other_x;
                new_w_tilde_sum += self.wtilde[id] * &other_x
            }
            (new_x, new_w_tilde_sum)
        };

        self.x = new_x;
        self.wtilde_sum_prev = new_w_tilde_sum;
        // self.x_prev = std::mem::replace(&mut self.x, new_x);
        // self.x = new_x;
        self.grad_prev = grad;
        self.sync.wait();
    }

    fn res(&self, res: &Arc<[AtomicF64]>) {
        res.get(self.k)
            .expect("do not exceed ITERATIONS + 1 length")
            .fetch_add((&self.x - &self.x_true).norm_squared(), Ordering::Relaxed);
    }
}
