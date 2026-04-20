// TODO:
// 1. figure out what a generic distributed convex optimization needs -> ie local loss function, local data, communication protocal
// 2. read a graph from a txt file and ensure that the graph has the right properties
// 3. generate local data, and share the data in optimization
// 4. generate output data and save to files -> python plotting.

use std::{
    ops::Sub,
    sync::{Arc, atomic::AtomicUsize, atomic::Ordering, mpsc},
    thread,
    time::Duration,
};

use atomic_float::AtomicF32;
use nalgebra::{self as na};
use rand::distr::{Bernoulli, Distribution};

const EPS: f32 = 0.0001;

struct Graph {
    adjacency_matrix: na::DMatrix<i8>,
}

impl Graph {
    fn degree_matrix(&self) -> na::DMatrix<i8> {
        let d = self.adjacency_matrix.column_sum();
        na::DMatrix::from_diagonal(&d)
    }

    fn laplacian(&self) -> na::DMatrix<i8> {
        let d = self.degree_matrix();
        d.sub(&self.adjacency_matrix)
    }

    fn weight_matrix(&self) -> na::DMatrix<f32> {
        let d = self.degree_matrix();
        let m = d.max();
        let tau = m as f32 + 1.0;
        let lap = d.sub(&self.adjacency_matrix).cast::<f32>();
        let w = na::DMatrix::identity(lap.nrows(), lap.ncols()) - lap / tau;
        assert_eq!(&w, &w.transpose(), "w must be symmetric");
        let ones = na::DVector::from_element(w.ncols(), 1.0);

        assert!(
            (&w * &ones - &ones).iter().all(|elem| elem.abs() < EPS),
            "W must be row stochastic"
        );
        w
    }
}

fn generate_graph(n: usize, threshold: f64) -> Graph {
    let mut adjacency_matrix = na::DMatrix::<i8>::zeros(n, n);
    let mut rng = rand::rng();
    let dist = Bernoulli::new(threshold).unwrap();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = if dist.sample(&mut rng) { 1 } else { 0 };
            adjacency_matrix[(i, j)] = v;
            adjacency_matrix[(j, i)] = v;
        }
    }
    return Graph { adjacency_matrix };
}

trait ObjectiveFunction {
    fn obj(&self, x: &na::DVector<f32>) -> f32;
    fn grad(&self, x: &na::DVector<f32>) -> na::DVector<f32>;
}

trait OptAlg {
    fn step(&mut self);
    fn res(&mut self, res: &Arc<[AtomicF32]>);
}

#[derive(Debug)]
struct LeastSquares {
    n: usize,
    p: usize,
    m: na::DMatrix<f32>,
    y: na::DVector<f32>,
    // true_x: na::DVector<f32>,
}

impl LeastSquares {
    fn new(n: usize, p: usize, mi: usize, true_x: &na::DVector<f32>) -> Self {
        assert_eq!(true_x.nrows(), p);

        let m = na::DMatrix::<f32>::new_random(mi, p);
        let noise = na::DVector::<f32>::new_random(mi);
        let y = &m * true_x + noise;

        Self { n, p, m, y }
    }
}

impl ObjectiveFunction for LeastSquares {
    fn obj(&self, x: &nalgebra::DVector<f32>) -> f32 {
        assert_eq!(x.nrows(), self.p);
        (&self.m * x - &self.y).norm() / 2.0 / self.n as f32
    }

    fn grad(&self, x: &nalgebra::DVector<f32>) -> nalgebra::DVector<f32> {
        assert_eq!(x.nrows(), self.p);
        &self.m.transpose() * (&self.m * x - &self.y) / self.n as f32
    }
}

#[derive(Debug)]
struct DGDNode<O: ObjectiveFunction, G: Fn(usize, f32) -> f32> {
    id: usize,
    n: usize,
    obj: O,
    x: na::DVector<f32>,
    x_true: na::DVector<f32>,
    w: na::DVector<f32>,
    k: usize,
    alpha_zero: f32,
    alpha_update: G,
    send: Box<[mpsc::Sender<(usize, na::DVector<f32>)>]>,
    recv: mpsc::Receiver<(usize, na::DVector<f32>)>,
    sync: Arc<AtomicUsize>,
}

unsafe impl<O: ObjectiveFunction, G: Fn(usize, f32) -> f32> Send for DGDNode<O, G> {}
unsafe impl<O: ObjectiveFunction, G: Fn(usize, f32) -> f32> Sync for DGDNode<O, G> {}

impl<O: ObjectiveFunction, G: Fn(usize, f32) -> f32 + Copy> DGDNode<O, G> {
    fn new(
        graph: Graph,
        mut objs: Vec<O>,
        x_0: na::DVector<f32>,
        x_true: na::DVector<f32>,
        alpha_zero: f32,
        alpha_update: G,
        sync: Arc<AtomicUsize>,
    ) -> Box<[Self]> {
        let n = graph.adjacency_matrix.nrows();
        assert!(objs.len() == n);

        let mut senders: Vec<Vec<mpsc::Sender<(usize, na::DVector<f32>)>>> = vec![Vec::new(); n];
        let mut receivers: Vec<mpsc::Receiver<(usize, na::DVector<f32>)>> = Vec::with_capacity(n);

        let w = graph.weight_matrix();

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

        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let node = DGDNode {
                id: n - i - 1,
                n,
                obj: objs.pop().expect("valid for all n"),
                // x_0: x_0.clone(),
                x: x_0.clone(),
                x_true: x_true.clone(),
                w: w.row(n - i - 1).clone().transpose(),
                k: 0,
                alpha_zero,
                alpha_update,
                send: senders.pop().expect("valid of n").into_boxed_slice(),
                recv: receivers.pop().expect("valid of n"),
                sync: sync.clone(),
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
            s.send((self.id, self.x.clone())).expect("can indeed send");
        }

        let mut new_x = self.w[self.id] * &self.x
            - (self.alpha_update)(self.k, self.alpha_zero) * self.obj.grad(&self.x);
        for _ in 0..self.send.len() {
            let (id, other_x) = self.recv.recv().expect("values are being sent");
            assert!(id != self.id);
            new_x += self.w[id] * other_x;
        }
        self.x = new_x;
        self.sync.fetch_add(1, Ordering::Relaxed);
    }

    fn res(&mut self, res: &Arc<[AtomicF32]>) {
        res.get(self.k)
            .expect("do not go beyong ITERATIONS len")
            .fetch_add((&self.x - &self.x_true).norm_squared(), Ordering::Relaxed);
    }
}

fn main() {
    let p = 4;
    let n = 10;
    let graph = generate_graph(n, 0.5);
    let x_0 = na::DVector::<f32>::zeros(p);
    let x_true = na::DVector::from_element(p, 300.0);
    let objs = (0..n)
        .map(|_| LeastSquares::new(n, p, 2, &x_true))
        .collect::<Vec<_>>();

    let sync = Arc::new(AtomicUsize::new(0));

    let nodes = DGDNode::new(
        graph,
        objs,
        x_0,
        x_true,
        0.5,
        |k, alpha| 3.0 * alpha / (k as f32).powf(1.0 / 3.0),
        Arc::clone(&sync),
    );

    const ITERATIONS: usize = 3000;
    let res = (0..ITERATIONS + 1)
        .map(|_| AtomicF32::new(0.0))
        .collect::<Arc<[_]>>();

    let mut handles = Vec::new();
    for mut node in nodes.into_iter() {
        let res_handle = Arc::clone(&res);
        handles.push(thread::spawn(move || {
            node.res(&res_handle);
            while node.k < ITERATIONS {
                while node.sync.load(Ordering::Relaxed) < node.k * node.n {
                    std::thread::sleep(Duration::from_nanos(1));
                }
                node.step();
                node.res(&res_handle);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("things went well");
    }

    let res0 = res
        .get(0)
        .expect("value exists")
        .load(Ordering::Relaxed)
        .sqrt();
    for i in 1..res.len() {
        let v = res
            .get(i)
            .expect("value exists")
            .load(Ordering::Relaxed)
            .sqrt()
            / res0;
        println!("k: {i}, v:{v}")
    }
}
