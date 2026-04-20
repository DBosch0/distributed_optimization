mod algorithms;
mod graph;
mod objectives;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use atomic_float::AtomicF64;
use nalgebra as na;

use algorithms::{OptAlg, dgd::DGDNode, extra::ExtraNode};
use graph::generate_graph;
use objectives::LeastSquares;

fn main() {
    let p = 5;
    let n = 10;
    let graph = generate_graph(n, 0.5);
    let x_0 = na::DVector::<f64>::zeros(p);
    let x_true = na::DVector::new_random(p);
    let objs = (0..n)
        .map(|_| LeastSquares::new(n, p, 10, &x_true))
        .collect::<Vec<_>>();

    // let nodes = DGDNode::new(graph, objs, x_0, x_true, 0.5, |k, alpha| {
    //     3.0 * alpha / (k as f64).powf(1.0 / 3.0)
    // });

    let nodes = ExtraNode::new(graph, objs, x_0, x_true, 0.5);

    const ITERATIONS: usize = 3000;
    let res: Arc<[AtomicF64]> = (0..=ITERATIONS).map(|_| AtomicF64::new(0.0)).collect();

    let mut handles = Vec::new();
    for mut node in nodes.into_iter() {
        let res_handle = Arc::clone(&res);
        handles.push(std::thread::spawn(move || {
            node.res(&res_handle);
            for _ in 0..ITERATIONS {
                node.step();
                node.res(&res_handle);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread completed successfully");
    }

    let res0 = res[0].load(Ordering::Relaxed).sqrt();
    for i in 1..res.len() {
        let v = res[i].load(Ordering::Relaxed).sqrt() / res0;
        println!("k: {i}, v: {v}");
    }
}
