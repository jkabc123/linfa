use linfa::{
    error::{Error, Result},
    Float, Label,
};
//ßßuse std::marker::PhantomData;

use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, Copy)]
pub enum ImpurityCriterion {
    Gini,
    Entropy,
    Mse,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Copy, Debug)]
///Hyperparameters for random forrest
pub struct RandomForrestParams {
    /// number of trees
    pub num_trees: usize,
    /// the minimum size for the leaf
    /// node of the tree
    pub min_leaf: usize,
    /// random generator
    pub seed: u64,
    /// percentage of sample comparing to training set
    pub sample_pct: f64,
    /// impurity criterion, gini or entropy for classfication
    /// mse for regression
    pub criterion: ImpurityCriterion,
}

impl RandomForrestParams {
    pub fn new() -> Self {
        RandomForrestParams {
            min_leaf: 5,
            num_trees: 10,
            seed: 42,
            sample_pct: 0.1,
            criterion: ImpurityCriterion::Gini,
        }
    }

    pub fn min_leaf(mut self, min_leaf: usize) -> Self {
        self.min_leaf = min_leaf;
        self
    }

    pub fn num_trees(mut self, num_trees: usize) -> Self {
        self.num_trees = num_trees;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn sample_pct(mut self, sample_pct: f64) -> Self {
        self.sample_pct = sample_pct;
        self
    }

    pub fn criterion(mut self, criterion: ImpurityCriterion) -> Self {
        self.criterion = criterion;
        self
    }
}
