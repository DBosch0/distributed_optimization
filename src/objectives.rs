use nalgebra as na;
use rand_distr::{Distribution, StandardNormal};

pub trait ObjectiveFunction {
    #[allow(unused)]
    fn obj(&self, x: &na::DVector<f64>) -> f64;
    fn grad(&self, x: &na::DVector<f64>) -> na::DVector<f64>;
}

#[derive(Debug)]
pub struct LeastSquares {
    n: usize,
    p: usize,
    m: na::DMatrix<f64>,
    y: na::DVector<f64>,
}

impl LeastSquares {
    pub fn new(n: usize, p: usize, mi: usize, true_x: &na::DVector<f64>) -> Self {
        assert_eq!(true_x.nrows(), p);
        let mut rng = rand::rng();
        let d = StandardNormal;

        let mut m = na::DMatrix::<f64>::zeros(mi, p);
        let mut noise = na::DVector::<f64>::zeros(mi);
        for i in 0..mi {
            noise[i] = d.sample(&mut rng);
            for j in 0..p {
                m[(i, j)] = d.sample(&mut rng);
            }
        }
        let y = &m * true_x + noise;
        Self { n, p, m, y }
    }
}

impl ObjectiveFunction for LeastSquares {
    fn obj(&self, x: &na::DVector<f64>) -> f64 {
        assert_eq!(x.nrows(), self.p);
        (&self.m * x - &self.y).norm_squared() / 2.0 / self.n as f64
    }

    fn grad(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        assert_eq!(x.nrows(), self.p);
        self.m.transpose() * (&self.m * x - &self.y) / self.n as f64
    }
}
