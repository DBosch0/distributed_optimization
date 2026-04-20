use nalgebra as na;

pub trait ObjectiveFunction {
    #[allow(unused)]
    fn obj(&self, x: &na::DVector<f32>) -> f32;
    fn grad(&self, x: &na::DVector<f32>) -> na::DVector<f32>;
}

#[derive(Debug)]
pub struct LeastSquares {
    n: usize,
    p: usize,
    m: na::DMatrix<f32>,
    y: na::DVector<f32>,
}

impl LeastSquares {
    pub fn new(n: usize, p: usize, mi: usize, true_x: &na::DVector<f32>) -> Self {
        assert_eq!(true_x.nrows(), p);
        let m = na::DMatrix::<f32>::new_random(mi, p);
        let noise = na::DVector::<f32>::new_random(mi);
        let y = &m * true_x + noise;
        Self { n, p, m, y }
    }
}

impl ObjectiveFunction for LeastSquares {
    fn obj(&self, x: &na::DVector<f32>) -> f32 {
        assert_eq!(x.nrows(), self.p);
        (&self.m * x - &self.y).norm_squared() / 2.0 / self.n as f32
    }

    fn grad(&self, x: &na::DVector<f32>) -> na::DVector<f32> {
        assert_eq!(x.nrows(), self.p);
        self.m.transpose() * (&self.m * x - &self.y) / self.n as f32
    }
}
