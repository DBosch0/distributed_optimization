use std::ops::Sub;

use nalgebra as na;
use rand::distr::{Bernoulli, Distribution};

const EPS: f64 = 1e-5;

pub struct Graph {
    pub adjacency_matrix: na::DMatrix<i8>,
}

impl Graph {
    fn degree_matrix(&self) -> na::DMatrix<i8> {
        let d = self.adjacency_matrix.column_sum();
        na::DMatrix::from_diagonal(&d)
    }

    #[allow(unused)]
    pub fn laplacian(&self) -> na::DMatrix<i8> {
        let d = self.degree_matrix();
        d.sub(&self.adjacency_matrix)
    }

    pub fn weight_matrix(&self) -> na::DMatrix<f64> {
        let d = self.degree_matrix();
        let m = d.max();
        let tau = m as f64 + 1.0;
        let lap = d.sub(&self.adjacency_matrix).cast::<f64>();
        let w = na::DMatrix::identity(lap.nrows(), lap.ncols()) - lap / tau;
        let ones = na::DVector::from_element(w.ncols(), 1.0f64);
        assert!(
            (&w - &w.transpose()).iter().all(|x| x.abs() < EPS),
            "W must be symmetric"
        );
        assert!(
            (&w * &ones - &ones).iter().all(|x| x.abs() < EPS),
            "W must be row stochastic"
        );
        w
    }
}

pub fn generate_graph(n: usize, threshold: f64) -> Graph {
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
    Graph { adjacency_matrix }
}
