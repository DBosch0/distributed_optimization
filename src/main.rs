// TODO:
// 1. figure out what a generic distributed convex optimization needs -> ie local loss function, local data, communication protocal
// 2. read a graph from a txt file and ensure that the graph has the right properties
// 3. generate local data, and share the data in optimization
// 4. generate output data and save to files -> python plotting.

use std::ops::Sub;

use nalgebra as na;
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

fn main() {
    let graph = generate_graph(4, 0.8);
    println!("{}", graph.adjacency_matrix);
    println!("{}", graph.laplacian());
    println!("{}", graph.weight_matrix());
}
