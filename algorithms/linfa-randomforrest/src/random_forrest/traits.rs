use linfa::Float;
use ndarray::{ArrayBase, Data, Ix2, Ix3, Axis, s, Slice, array};

use crate::debug;

pub trait GiniImpurity<F: Float> {
    fn gini_score(&self) -> F;
}

/// shape (n_nodes, n_distinct_labels, n_targets)
impl<F: Float, D> GiniImpurity<F> for ArrayBase<D, Ix3>
where
    D: Data<Elem = F>,
{
    fn gini_score(&self) -> F {
        //debug!(self);
        //debug!(self.sum_axis(Axis(1)));
        
        let p = self / & (self.sum_axis(Axis(1)).insert_axis(Axis(1)));
        //debug!(p.clone());
        let p2 = &p * &p;
        let sum_p2 = p2.sum_axis(Axis(1));
        
        let gi = &-sum_p2 + &array![F::from(1).unwrap()];
        //debug!(gi.clone());
        //let m = self.sum_axis(Axis(1)) / self.sum();
        let mi = self.sum_axis(Axis(1));
        let m = &mi / &mi.sum_axis(Axis(0));
        // debug!(mi.clone());
        // debug!(m.clone());
        // debug!(self.sum_axis(Axis(1)));
        // debug!(self.sum_axis(Axis(1)) / self.sum_axis(Axis(1)).sum_axis(Axis(0)));
        let g = (gi * &m).sum();
        g
    }
}

pub fn gini_score_3d<D, F> (x: &ArrayBase<D, Ix3>) -> F
where
    F: Float, 
    D: Data<Elem = F> 
{
    let p = x / &(x.sum_axis(Axis(1)).insert_axis(Axis(1)));
    //debug!(p.clone());
    let p2 = &p * &p;
    let sum_p2 = p2.sum_axis(Axis(1));
    
    let gi = &-sum_p2 + &array![F::from(1).unwrap()];
    //debug!(gi.clone());
    //let m = self.sum_axis(Axis(1)) / self.sum();
    let mi = x.sum_axis(Axis(1));
    let m = &mi / &mi.sum_axis(Axis(0));
    // debug!(mi.clone());
    // debug!(m.clone());
    // debug!(self.sum_axis(Axis(1)));
    // debug!(self.sum_axis(Axis(1)) / self.sum_axis(Axis(1)).sum_axis(Axis(0)));
    let g = (gi * &m).sum();
    g
    //F::from(0.0).unwrap()
}

pub fn gini_score_2d<D, F> (x: &ArrayBase<D, Ix2>) -> F
where
    F: Float,
    D: Data<Elem = F>
{
    let p = x / & (x.sum_axis(Axis(0)).insert_axis(Axis(0)));
    //debug!(p.clone());
    let p2 = &p * &p;
    let sum_p2 = p2.sum_axis(Axis(0));
    
    let gi = &-sum_p2 + &array![F::from(1).unwrap()];
    gi.sum()

}

/// shape (n_distinct_labels, n_targets)
impl<F: Float, D> GiniImpurity<F> for ArrayBase<D, Ix2>
where
    D: Data<Elem = F>,
{
    fn gini_score(&self) -> F {
        //debug!(self);
        //debug!(self.sum_axis(Axis(0)));
        let p = self / & (self.sum_axis(Axis(0)).insert_axis(Axis(0)));
        //debug!(p.clone());
        let p2 = &p * &p;
        let sum_p2 = p2.sum_axis(Axis(0));
        
        let gi = &-sum_p2 + &array![F::from(1).unwrap()];
        gi.sum()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    use super::debug;
    use super::GiniImpurity;
    use super::gini_score_3d;
    use ndarray::{arr2, arr3, stack, Axis};
    use serde_json::{Result, Value};
   

    #[test]
    fn test_gini_dim3() {
        // shape (2, 3, 1)
        let a = array![[[30.0], [30.0], [30.0]],
                                                     [[30.0], [40.0], [50.0]]];
        assert_abs_diff_eq!(a.gini_score(),0.658730158, epsilon=0.1);
        assert_abs_diff_eq!(a.view().gini_score(),0.658730158, epsilon=0.1);
        assert_abs_diff_eq!((&a).gini_score(),0.658730158, epsilon=0.1);
    }

    #[test]
    fn test_gini_dim3_new() {
        // shape (2, 3, 1)
        let mut a = array![[[30.0], [30.0], [30.0]],
                                                     [[30.0], [40.0], [50.0]]];
        assert_abs_diff_eq!(gini_score_3d(&a),0.658730158, epsilon=0.1);
        assert_abs_diff_eq!(gini_score_3d(&a.view()),0.658730158, epsilon=0.1);
        assert_abs_diff_eq!(gini_score_3d(&a.view_mut()),0.658730158, epsilon=0.1);
    }

    #[test]
    fn test_gini_dim3_multitargets() {
        // shape (2, 3, 1)
        let a = array![[[30.0, 30.0], [30.0, 30.0], [30.0, 30.0]],
                                                     [[30.0, 30.0], [40.0, 40.0], [50.0, 50.0]]];
        assert_abs_diff_eq!(a.gini_score(),0.658730158*2.0, epsilon=0.1);
        //debug!(a.gini_score());
    }

    #[test]
    fn test_gini_dim2() {
        // shape (3, 1)
        let a = array![[30.0], [40.0], [50.0]];
        assert_abs_diff_eq!(a.gini_score(),0.658730158, epsilon=0.1);
        
    }

    #[test]
    fn test_gini_dim2_multitargets() {
        // shape (3, 2)
        let a = array![[30.0,30.0], [40.0,40.0], [50.0,50.0]];
        assert_abs_diff_eq!(a.gini_score(),0.658730158*2.0, epsilon=0.1);
    }

    #[test]
    fn test_gini_zero(){
        let a = array![[[5.0],[0.0],[0.0]], [[0.0],[0.0],[0.0]]];
        debug!(a.clone());
        debug!(a.shape());
        debug!(a.gini_score());
    }

    #[test]
    fn test_stack(){
        let a = arr2(&[[2., 2.], [3., 3.]]);
        let a = a.insert_axis(Axis(0));
        debug!(a.shape());
        debug!(stack![Axis(0), a, a]);
    }

    #[test]
    fn test_serde(){
        
    }

    
}
