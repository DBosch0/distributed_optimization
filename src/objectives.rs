use nalgebra as na;
use rand_distr::{Distribution, StandardNormal};

pub trait ObjectiveFunction {
    #[allow(unused)]
    fn obj(&self, x: &na::DVector<f64>) -> f64;
    fn grad(&self, x: &na::DVector<f64>) -> na::DVector<f64>;
}

#[derive(Debug)]
pub struct LeastSquares {
    n: usize, //number of agents
    m: usize, // number of samples per agent
    p: usize, //dimension of samples
    x: na::DMatrix<f64>,
    y: na::DVector<f64>,
}

impl LeastSquares {
    pub fn new(n: usize, p: usize, mi: usize, true_x: &na::DVector<f64>) -> Self {
        assert_eq!(true_x.nrows(), p);
        let mut rng = rand::rng();
        let d = StandardNormal;

        let mut x = na::DMatrix::<f64>::zeros(mi, p);
        let mut noise = na::DVector::<f64>::zeros(mi);
        for i in 0..mi {
            noise[i] = d.sample(&mut rng);
            for j in 0..p {
                x[(i, j)] = d.sample(&mut rng);
            }
        }
        let y = &x * true_x / (p as f64).sqrt() + 0.1f64.sqrt() * noise;
        Self { n, m: mi, p, x, y }
    }
}

impl ObjectiveFunction for LeastSquares {
    fn obj(&self, x: &na::DVector<f64>) -> f64 {
        assert_eq!(x.nrows(), self.p);
        (&self.x * x / (self.p as f64).sqrt() - &self.y).norm_squared() / 2.0 / (self.m) as f64
    }

    fn grad(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        assert_eq!(x.nrows(), self.p);
        self.x.transpose() * (&self.x * x / (self.p as f64).sqrt() - &self.y)
            / (self.m) as f64
            / (self.p as f64).sqrt()
    }
}
