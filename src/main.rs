// TODO:
// 1. figure out what a generic distributed convex optimization needs -> ie local loss function, local data, communication protocal
// 2. read a graph from a txt file and ensure that the graph has the right properties
// 3. generate local data, and share the data in optimization
// 4. generate output data and save to files -> python plotting.

use std::ops::Sub;

use nalgebra::{self as na};
use rand::distr::{Bernoulli, Distribution};
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
        // assert_eq!(w, w.transpose(), "w must be symmetric");
        // let ones = na::DVector::from_element(w.ncols(), 1.0);
        // assert_eq!(&w * &ones, ones, "W must be row stochastic");
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
    fn obj(&self, x: &na::DMatrix<f32>) -> f32;
    fn grad(&self, x: &na::DMatrix<f32>) -> na::DMatrix<f32>;
}

struct LeastSquares {
    n: usize,
    p: usize,
    ms: Vec<na::DMatrix<f32>>,
    ys: Vec<na::DVector<f32>>,
    // true_x: na::DVector<f32>,
}

impl LeastSquares {
    fn new(n: usize, p: usize, mis: &[usize], true_x: &na::DVector<f32>) -> Self {
        // assert_eq!(true_x.nrows(), n);
        assert_eq!(true_x.nrows(), p);
        assert_eq!(mis.len(), n);

        let mut ms = Vec::new();
        let mut ys = Vec::new();
        for (i, mi) in mis.into_iter().enumerate() {
            ms.push(na::DMatrix::<f32>::new_random(*mi, p));
            let noise = na::DVector::<f32>::new_random(*mi);
            ys.push(&ms[i] * true_x + noise);
        }

        Self {
            n,
            p,
            ms,
            // true_x,
            ys,
        }
    }
}

impl ObjectiveFunction for LeastSquares {
    fn obj(&self, x: &nalgebra::DMatrix<f32>) -> f32 {
        assert_eq!(x.nrows(), self.n);
        assert_eq!(x.ncols(), self.p);

        let mut s = 0.0;
        for i in 0..self.n {
            s += (&self.ms[i] * &x.row(i).transpose() - &self.ys[i]).norm();
        }

        s / 2.0 / self.n as f32
    }

    fn grad(&self, x: &nalgebra::DMatrix<f32>) -> nalgebra::DMatrix<f32> {
        assert_eq!(x.nrows(), self.n);
        assert_eq!(x.ncols(), self.p);
        let mut out = na::DMatrix::zeros(x.nrows(), x.ncols());
        for i in 0..self.n {
            let row = &self.ms[i].transpose() * (&self.ms[i] * &x.row(i).transpose() - &self.ys[i]);
            out.set_row(i, &row.transpose())
        }
        out / self.n as f32
    }
}

struct DGD<F: ObjectiveFunction, G>
where
    G: Fn(f32, usize) -> f32,
{
    w: na::DMatrix<f32>,
    x_0: na::DMatrix<f32>,
    x: na::DMatrix<f32>,
    x_true: na::DMatrix<f32>,
    alpha_zero: f32,
    k: usize,
    f: F,
    alpha_update_rule: G,
}

impl<F: ObjectiveFunction, G: Fn(f32, usize) -> f32> DGD<F, G> {
    fn new(
        graph: Graph,
        x_0: na::DMatrix<f32>,
        x_true: na::DMatrix<f32>,
        alpha_zero: f32,
        f: F,
        alpha_update_rule: G,
    ) -> Self {
        Self {
            w: graph.weight_matrix(),
            x_0: x_0.clone(),
            x: x_0,
            x_true,
            alpha_zero,
            k: 0,
            f,
            alpha_update_rule,
        }
    }

    fn step(&mut self) {
        self.k += 1;
        let alpha = (self.alpha_update_rule)(self.alpha_zero, self.k);
        self.x = &self.w * &self.x - alpha * self.f.grad(&self.x);
    }

    fn res(&self) -> f32 {
        (&self.x - &self.x_true).norm() / (&self.x_0 - &self.x_true).norm()
    }
}

fn main() {
    let p = 5;
    let n = 10;
    let graph = generate_graph(n, 0.5);
    let x_0 = na::DMatrix::zeros(n, p);
    let x_true = na::DMatrix::from_element(n, p, 300.0);
    // let x_true = na::DVector::from_element(p, 300.0);
    let ls = LeastSquares::new(n, p, &vec![1; n], &x_true.row(0).transpose());
    println!("{}", ls.grad(&na::DMatrix::from_element(n, p, 0.0)));

    let mut dgd = DGD::new(graph, x_0, x_true, 0.5276, ls, |alpha_0, k| {
        3.0 * alpha_0 / (k as f32).powf(1.0 / 3.0)
    });
    for k in 1..=3000 {
        dgd.step();
        println!("{}:{}", k, dgd.res());
    }
}
