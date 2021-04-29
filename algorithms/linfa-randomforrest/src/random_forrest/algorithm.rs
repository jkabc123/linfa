//! Random Forrest
//!
#![deny(missing_docs)]
use std::{collections::HashMap, hash::Hash};

use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix3, OwnedRepr, Slice, array, s, stack};

use crate::ImpurityCriterion;

use super::hyperparameters::RandomForrestParams;
use super::traits::{GiniImpurity, gini_score_3d, gini_score_2d};
use crate::debug;

use linfa::{dataset::Labels, traits::*, DatasetBase, Float, Label};

use ndarray_rand::rand::{seq, Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize, Serializer, Deserializer};
use num_traits::{cast::ToPrimitive};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
///Random Forrest Classifier
pub struct RandomForrestClassifer<F:Float, L:Hash+Eq> {
    ///the decison trees of the random forrest classifier
    pub trees: Vec<DecisionTreeClassifier<F>>,

    ///mappings from label to integer
    mapping: HashMap<L, usize>,

    ///sorted labels
    labels: Vec<L>,
}


fn float_serialize<S, F>(x: &F, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    F: Float + ToPrimitive
{
    if x.is_infinite() {
        s.serialize_str("Infinity")
    }else{
        s.serialize_str(&x.to_f64().unwrap().to_string())
    }
}

fn float_deserialize<'de, D, F>(deserializer: D) -> Result<F, D::Error>
where
    D: Deserializer<'de>,
    F: Float
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    if s=="Infinity" {
        Ok(F::infinity())
    }else{
        let f = s.parse::<f64>().unwrap();
        Ok(F::from(f).unwrap())
    }

}

///Random Forrest Regressor
pub struct RandomForrestRegressor<F> {
    /// the decison trees of the random forrest regressor
    trees: Vec<DecisionTreeRegressor<F>>,
}

#[derive(Debug)]
///Base Decision Tree
pub enum DecisionTree<F> {
    ///empty tree
    Empty,
    ///non empty tree
    NonEmpty(Box<DecisionTreeNode<F>>),
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
///Base Decision Tree
pub enum DecisionTreeClassifier<F:Float> {
    ///empty tree
    Empty,
    ///non empty tree
    NonEmpty(Box<DecisionTreeClassifierNode<F>>),
}

#[derive(Debug)]
///Decision Tree Regressor
pub struct DecisionTreeRegressor<F>(DecisionTree<F>);

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
///Type for Tree node value
pub enum NodeValue<F> {
    ///value for Classifier
    ClassifierNode(HashMap<usize, f32>),
    ///value for Regressor
    RegressorNode(F),
}

#[derive(Debug)]
///Decision tree node
pub struct DecisionTreeNode<F> {
    /// list of indexes
    idxs: Vec<usize>,
    ///size of the dataset of the tree
    size: usize,
    ///value of the tree
    value: Option<NodeValue<F>>,
    //value: Option<Array1<F>>,
    ///score of the tree
    score: F,
    ///the index of the split feature
    split_idx: Option<usize>,
    ///the value to split the feature
    split_val: Option<F>,
    ///left child tree
    left: DecisionTree<F>,
    ///right child tree
    right: DecisionTree<F>,
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
///Decision tree node
pub struct DecisionTreeClassifierNode<F:Float> {
    /// value of the tree
    /// shape is (n_distinct_labels, n_targets)
    value: Option<Array2<F>>,
    ///score of the tree
    #[serde(serialize_with = "float_serialize")]
    #[serde(deserialize_with = "float_deserialize")]
    score: F,
    ///the index of the split feature
    split_idx: Option<usize>,
    ///the value to split the feature
    split_val: Option<F>,
    ///left child tree
    left: DecisionTreeClassifier<F>,
    ///right child tree
    right: DecisionTreeClassifier<F>,
    ///depth
    depth: usize,
}

impl<F: Float, L: Label> RandomForrestClassifer<F, L> {
    ///Constructs default parameters
    pub fn params() -> RandomForrestParams {
        RandomForrestParams::new()
    }
}

impl<F: Float + PartialOrd, L: Label + Debug>
    Predict<DatasetBase<Array2<F>, Array2<L>>, DatasetBase<Array2<F>, Array2<L>>>
    for RandomForrestClassifer<F, L>
{
    /// input data record shape (n_rows, n_features)
    /// output data target shape (n_rows, n_targets)
    fn predict(
        &self,
        data: DatasetBase<Array2<F>, Array2<L>>,
    ) -> DatasetBase<Array2<F>, Array2<L>> {
        // a shape (n_trees, n_rows, n_distinct_labels, n_targets)
        let mut preds = Array4::zeros((
            self.trees.len(),          // n_trees
            data.records().shape()[0], // n_rows
            self.labels.len(),         // n_distinct_labels
            data.targets().shape()[1], // n_targets
        ));

        for (i, tree) in self.trees.iter().enumerate() {
            // shape (n_rows, n_distinct_labels, n_targets)
            let b = tree.predict(
                data.records.view(),
                self.labels.len(),
                data.targets().shape()[1],
            );
            //debug!(b.clone());
            preds.slice_mut(s![i, .., .., ..]).assign(&b);
        }

        let preds = preds.mean_axis(Axis(0)).unwrap();
        //debug!(a.clone());

        // map along label axis and returns the label with the largest frequencies
        let preds = preds.map_axis(Axis(1), |r| {
            let max = r
                .into_iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();

            for x in r.into_iter().enumerate() {
                if x.1 == max {
                    return self.labels.get(x.0).unwrap().clone();
                }
            }

            panic!("It shouldn't reach here");
        });

        DatasetBase::new(data.records, preds)
    }
}

impl<F: Float, L: Label> Predict<Array2<F>, Array2<F>> for RandomForrestClassifer<F, L> {
    fn predict(&self, data: Array2<F>) -> Array2<F> {
        let mut a = Array3::zeros((self.trees.len(), data.shape()[0], self.mapping.keys().len()));
        for (i, tree) in self.trees.iter().enumerate() {
            let b = tree.predict(data.view(), self.mapping.keys().len(), self.labels.len());
            a.slice_mut(s![i, .., ..]).assign(&b);
        }

        a.mean_axis(Axis(0)).unwrap()
    }
}

/// create a mapping between label and usize
/// the labels will be sorted in ascending order first
/// so that the mapped usize will keep the same order as the label
fn create_mapping<L: Label + Ord>(labels: &Vec<L>) -> HashMap<L, usize> {
    let mut l = labels.clone();
    l.sort_unstable();
    let mapping = l
        .into_iter()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect::<HashMap<L, usize>>();
    return mapping;
}

impl<F: Float> RandomForrestRegressor<F> {
    /// Construct default parameters
    pub fn params() -> RandomForrestParams {
        RandomForrestParams::new()
    }
}

impl<'a, F: Float, L: Label + Copy + Debug + Ord + 'a> Fit<'a, Array2<F>, Array2<L>>
    for RandomForrestParams
{
    type Object = RandomForrestClassifer<F, L>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, Array2<L>>) -> Self::Object {
        let mapping = create_mapping(&dataset.targets.labels());
        let mut rng = Isaac64Rng::seed_from_u64(self.seed);
        let trees = (0..self.num_trees)
            .into_iter()
            .map(|_| self.create_tree(dataset, &mut rng, &mapping))
            .collect();
        let mut labels = dataset.targets.labels();
        labels.sort_unstable();

        RandomForrestClassifer {
            trees,
            mapping,
            labels,
        }
    }
}

impl<'a, F: Float> Fit<'a, Array2<F>, Array1<F>> for RandomForrestParams {
    type Object = RandomForrestRegressor<F>;

    fn fit(&self, dataset: &DatasetBase<Array2<F>, Array1<F>>) -> Self::Object {
        let mut rng = Isaac64Rng::seed_from_u64(self.seed);
        let trees = (0..self.num_trees)
            .into_iter()
            .map(|_| self.create_tree_reg(dataset, &mut rng))
            .collect();
        RandomForrestRegressor { trees: trees }
    }
}

impl RandomForrestParams {
    /// Create a new classifier tree
    fn create_tree<F: Float, L: Label + Copy + Debug + Ord>(
        &self,
        dataset: &DatasetBase<Array2<F>, Array2<L>>,
        rng: &mut impl Rng,
        mapping: &HashMap<L, usize>,
    ) -> DecisionTreeClassifier<F> {
        let full_sz = dataset.records().nrows();
        let sample_sz = (full_sz as f64 * self.sample_pct) as usize;

        let idxs = seq::index::sample(rng, full_sz, sample_sz).into_vec();
        let mut sorted_labels = dataset.labels();
        sorted_labels.sort_unstable();
        DecisionTreeClassifier::new(
            &sorted_labels,
            idxs,
            self,
            &DatasetBase::new(dataset.records(), dataset.targets()),
            mapping,
            0,
        )
    }

    /// Create a new regressor tree
    fn create_tree_reg<F: Float>(
        &self,
        dataset: &DatasetBase<Array2<F>, Array1<F>>,
        rng: &mut impl Rng,
    ) -> DecisionTreeRegressor<F> {
        let full_sz = dataset.records().nrows();
        let sample_sz = (full_sz as f64 * self.sample_pct) as usize;

        //let num_features = dataset.records.shape()[1];
        let idxs = seq::index::sample(rng, full_sz, sample_sz).into_vec();
        DecisionTreeRegressor::new(idxs, sample_sz, self, dataset)
    }
}

impl<F: Float> DecisionTreeClassifier<F> {
    /// Construct and fit the tree
    fn new<L: Label + Copy + Debug + Ord>(
        sorted_labels: &Vec<L>,
        idxs: Vec<usize>,
        params: &RandomForrestParams,
        dataset: &DatasetBase<&Array2<F>, &Array2<L>>,
        mapping: &HashMap<L, usize>,
        depth: usize,
    ) -> Self {
        let mut tree = DecisionTreeClassifier::NonEmpty(Box::new(DecisionTreeClassifierNode {
            value: None,
            score: F::infinity(),
            //score: F::from(1_000.0).unwrap(),
            split_idx: None,
            split_val: None,
            // split_idx: Some(0),
            // split_val: Some(F::from(0.0).unwrap()),
            left: DecisionTreeClassifier::Empty,
            right: DecisionTreeClassifier::Empty,
            depth,
        }));
        if idxs.len() != 0 {
            tree.fit(dataset, params, mapping, idxs, sorted_labels);
        }
        tree
    }

    /// Get the frequencies of the labels in the ascending order of the labels
    fn collect_freqs<L: Label + Copy + Debug + Ord>(
        &self,
        label_freqs: HashMap<L, f32>,
        sorted_labels: &Vec<L>,
    ) -> Array1<F> {
        let mut freqs = Array1::zeros(sorted_labels.len());
        for (i, label) in sorted_labels.iter().enumerate() {
            // if label is missing, the frequency is 0.0
            freqs[[i]] = F::from_f32(*label_freqs.get(label).unwrap_or(&0.0)).unwrap();
        }
        freqs
    }

    /// Return target's label frequencies
    /// Input targets shape is (n_samples, n_targets)
    /// Output shape is (n_distinct_labels, n_targets)
    fn label_frequencies(&mut self, targets: ArrayView2<usize>, n_labels: usize) -> Array2<F> {
        //debug!(targets);
        let mut freqs = Array2::zeros((n_labels, targets.ncols()));
        for (i, target) in targets.genrows().into_iter().enumerate() {
            for (j, col) in target.gencolumns().into_iter().enumerate() {
                for row in col.into_iter() {
                    freqs[[*row, j]] += F::from(1.).unwrap();
                }
            }
        }
        freqs
    }

    /// Fit the tree based on the dataset
    /// Input dataset record shape (n_samples, n_features)
    /// Input dataset target shape (n_samples, n_targets)
    fn fit<L: Label + Copy + Debug + Ord>(
        &mut self,
        dataset: &DatasetBase<&Array2<F>, &Array2<L>>,
        params: &RandomForrestParams,
        mapping: &HashMap<L, usize>,
        idxs: Vec<usize>,
        sorted_labels: &Vec<L>,
    ) {
        let n_labels = sorted_labels.len();
        let x = dataset.records().select(Axis(0), &idxs);
        let y = dataset.targets().select(Axis(0), &idxs);

        let y_mapped = y.map(|x| *mapping.get(x).unwrap());

        // shape is (n_rows, n_target)
        let ds_mapped = DatasetBase::new(x, y_mapped);
        let ds_ori = DatasetBase::new(ds_mapped.records(), &y);

        // shape is (n_distinct_labels, n_targets)
        let freq = self.label_frequencies(ds_mapped.targets().view(), sorted_labels.len());
        //debug!(freq.gini_score());

        if let DecisionTreeClassifier::NonEmpty(node) = self {
            let x = ds_mapped.records();
            let y = ds_mapped.targets();
            node.value = Some(freq.clone());
            // if the gini score is below the threshold, stop the split process
            //if freq.gini_score() < F::from(0.6).unwrap() {

            // if gini_score_2d(&freq) < F::from(0.6).unwrap() {
            if gini_score_2d(&freq) == F::from(0.0).unwrap() {  
                //debug!(freq.gini_score());
                return;
            }

            let c = x.shape()[1];
            for i in 0..c {
                // x shape is (sample_size, 1)
                let x = x.slice_axis(Axis(1), Slice::from(i..i + 1));
                self.find_best_split(x.view(), y.view(), i, n_labels, params);
            }
        }

        // fit left and right
        let (left_idxs, right_idxs) = self.find_left_right_idxs(ds_ori.records().view());
        if left_idxs.is_empty() && right_idxs.is_empty() {
            return;
        }

        if let DecisionTreeClassifier::NonEmpty(node) = self {
            // if node.depth+1 > 3 {
            //   return;
            // }
            node.left = DecisionTreeClassifier::new(
                sorted_labels,
                left_idxs,
                params,
                &ds_ori,
                mapping,
                node.depth + 1,
            );
            node.right = DecisionTreeClassifier::new(
                sorted_labels,
                right_idxs,
                params,
                &ds_ori,
                mapping,
                node.depth + 1,
            );
        }
    }

    fn find_left_right_idxs(&self, dataset: ArrayView2<F>) -> (Vec<usize>, Vec<usize>) {
        let mut left: Vec<usize> = Vec::new();
        let mut right: Vec<usize> = Vec::new();
        if let DecisionTreeClassifier::NonEmpty(node) = self {
            if node.score == F::infinity() {
                return (left, right);
            }
            let split = node.split_idx.unwrap();
            let val = node.split_val.unwrap();

            for idx in 0..dataset.nrows() {
                if dataset[[idx, split]] <= val {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }
        }
        (left, right)
    }

    ///update the tree based on gini impurity
    fn update_gini_split(
        &mut self,
        sort_x: ArrayView2<F>,
        sort_y: ArrayView2<usize>,
        feature_idx: usize,
        params: &RandomForrestParams,
    ) {
        if let DecisionTreeClassifier::NonEmpty(node) = self {
            let sz = sort_y.shape()[0];
            let mut r_cnt = sz as f64; //F::from(sz).unwrap();
            let mut l_cnt = 0 as f64; //F::from(0).unwrap();
            let n_targets = sort_y.shape()[1];

            let mut r_vec: Vec<HashMap<usize, usize>> = Vec::with_capacity(sz);
            let mut l_vec: Vec<HashMap<usize, usize>> = Vec::with_capacity(sz);
            for (j, yj) in sort_y.gencolumns().into_iter().enumerate() {
                let l_ent = HashMap::new();
                l_vec.push(l_ent);
                let mut r_ent = HashMap::new();
                for xi in yj.into_iter() {
                    let counter = r_ent.entry(*xi).or_insert(0 as usize);
                    *counter += 1;
                }
                r_vec.push(r_ent);
            }

            for i in 0..(sz - params.min_leaf) {
                let xi = sort_x[[i, 0]];
                let xi_plus1 = sort_x[[i + 1, 0]];

                for j in 0..n_targets {
                    r_cnt -= 1.;
                    l_cnt += 1.;
                    let l = &mut l_vec[j];
                    let r = &mut r_vec[j];
                    let d = sort_y[[i, j]];
                    //need to check this line
                    let counter = r.get_mut(&d).unwrap(); //.entry(d).or_insert(0 as usize);
                    *counter -= 1;
                    if *counter == 0 {
                        r.remove_entry(&d);
                    }
                    let counter = l.entry(d).or_insert(0 as usize);
                    *counter += 1;
                }

                if i < params.min_leaf - 1 || xi == xi_plus1 {
                    continue;
                }

                let current_score = gini_score(&mut l_vec, &mut r_vec, n_targets, l_cnt, r_cnt);

                if feature_idx == 2 && xi == F::from(1.9).unwrap() {
                    //&& xi <= F::from(2.45).unwrap() {
                    println!("xi: {:?}, {:?}", xi, i);
                    println!("x_i+1 : {:?}", xi_plus1);
                    println!("l_vec: {:?}", l_vec);
                    println!("r_vec: {:?}", r_vec);
                    println!("score: {:?}", current_score);
                    println!("node value: {:?}", node.value);
                }
                if current_score < node.score {
                    node.score = current_score;
                    node.split_idx = Some(feature_idx);
                    node.split_val = Some((xi + xi_plus1) / F::from(2.0).unwrap());
                }
            }
        }
    }

    ///update the tree based on gini impurity
    fn update_tree_gini(
        &mut self,
        sort_x: ArrayView2<F>,
        sort_y: ArrayView2<usize>,
        feature_idx: usize,
        n_labels: usize,
        params: &RandomForrestParams,
    ) {
        if let DecisionTreeClassifier::NonEmpty(node) = self {
            let sz = sort_y.shape()[0];
            let n_targets = sort_y.shape()[1];

            // shape (2, n_distinct_labels, n_targets)
            // 0 is left, 1 is right
            let mut left_right: Array3<F> = Array3::zeros((2, n_labels, n_targets));

            for (j, target_j) in sort_y.gencolumns().into_iter().enumerate() {
                for d in target_j.into_iter(){
                    left_right[[1, *d, j]] += F::from(1.0).unwrap();
                }
            }

            for i in 0..(sz - params.min_leaf) {
                let xi = sort_x[[i, 0]];
                let xi_plus1 = sort_x[[i + 1, 0]];

                for j in 0..n_targets {
                    let d = sort_y[[i,j]];
                    if left_right[[1, d, j]] != F::from(0.0).unwrap() {left_right[[1, d, j]] -= F::from(1.0).unwrap();}
                    left_right[[0, d, j]] += F::from(1.0).unwrap();

                }

                if i < params.min_leaf - 1 || xi == xi_plus1 {
                    continue;
                }

                let current_score = left_right.gini_score();
                
                if current_score < node.score {
                    node.score = current_score;
                    node.split_idx = Some(feature_idx);
                    node.split_val = Some((xi + xi_plus1) / F::from(2.0).unwrap());
                }
            }
        }
    }

    ///update the tree based on their score
    fn update_tree<Score>(
        &mut self,
        sort_x: ArrayView2<F>,
        sort_y: ArrayView2<usize>,
        feature_idx: usize,
        n_labels: usize,
        params: &RandomForrestParams,
        score_fn: Score
    ) 
        where Score: Fn(&ArrayBase<OwnedRepr<F>, Ix3>)->F,
              //D: Data<Elem = F>
    {
        if let DecisionTreeClassifier::NonEmpty(node) = self {
            let sz = sort_y.shape()[0];
            let n_targets = sort_y.shape()[1];

            // shape (2, n_distinct_labels, n_targets)
            // 0 is left, 1 is right
            let mut left_right: Array3<F> = Array3::zeros((2, n_labels, n_targets));

            for (j, target_j) in sort_y.gencolumns().into_iter().enumerate() {
                for d in target_j.into_iter(){
                    left_right[[1, *d, j]] += F::from(1.0).unwrap();
                }
            }

            for i in 0..(sz - params.min_leaf) {
                let xi = sort_x[[i, 0]];
                let xi_plus1 = sort_x[[i + 1, 0]];

                for j in 0..n_targets {
                    let d = sort_y[[i,j]];
                    if left_right[[1, d, j]] != F::from(0.0).unwrap() {left_right[[1, d, j]] -= F::from(1.0).unwrap();}
                    left_right[[0, d, j]] += F::from(1.0).unwrap();

                }

                if i < params.min_leaf - 1 || xi == xi_plus1 {
                    continue;
                }

                let current_score = score_fn(&left_right);//left_right.gini_score();
                
                if current_score < node.score {
                    node.score = current_score;
                    node.split_idx = Some(feature_idx);
                    node.split_val = Some((xi + xi_plus1) / F::from(2.0).unwrap());
                }
            }
        }
    }


    /// find the best split for a feature
    fn find_best_split(
        &mut self,
        // dataset: &DatasetBase<Array2<F>, Array2<L>>,
        x: ArrayView2<F>,
        y: ArrayView2<usize>,
        feature_idx: usize,
        n_labels: usize,
        params: &RandomForrestParams,
    ) {
        // sort by x's value and returns sorted idx list
        let mut sort_idx: Vec<(usize, _)> = x
            //.slice(s![.., feature_idx])
            .into_iter()
            .enumerate()
            .collect();
        sort_idx.sort_unstable_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let sort_idx: Vec<usize> = sort_idx.into_iter().map(|a| a.0).collect();

        // sort_x shape (sample_size, 1)
        let sort_x = x.select(Axis(0), &sort_idx);
        // sort_y shape (sample_size, label size)
        let sort_y = y.select(Axis(0), &sort_idx);
        match params.criterion {
            ImpurityCriterion::Gini => {
                //self.update_gini_split(sort_x.view(), sort_y.view(), feature_idx, params)
                //self.update_tree_gini(sort_x.view(), sort_y.view(), feature_idx, n_labels, params)
                self.update_tree(sort_x.view(), sort_y.view(), feature_idx, n_labels, params, gini_score_3d)
            }
            ImpurityCriterion::Entropy => {
                self.update_gini_split(sort_x.view(), sort_y.view(), feature_idx, params)
            }
            _ => self.update_gini_split(sort_x.view(), sort_y.view(), feature_idx, params),
        }
    }

    /// input x shape (n_rows, n_features)
    /// output shape (n_rows, n_distinct_labels, n_targets)
    fn predict(&self, x: ArrayView2<F>, n_labels: usize, n_targets: usize) -> Array3<F> {
        //debug!(n_targets);
        // shape (n_rows, n_distinct_labels, n_targets)
        let mut preds = Array3::zeros((x.shape()[0], n_labels, n_targets));
        for (i, row) in x.genrows().into_iter().enumerate() {
            // shape (n_distinct_labels, n_targets)
            let pred = self.predict_row(row.view());
            preds.slice_mut(s![i, .., ..]).assign(&pred);
        }
        //debug!(preds.clone());
        preds
    }

    /// input x shape (n_features)
    /// output shape (n_distinct_labels, n_targets)
    fn predict_row(&self, x: ArrayView1<F>) -> Array2<F> {
        match self {
            DecisionTreeClassifier::NonEmpty(node) => {
                if let DecisionTreeClassifier::Empty = node.left {
                    if let DecisionTreeClassifier::Empty = node.right {
                        let x = node.value.as_ref().unwrap().clone();
                        //println!("x: {:?}", x);
                        // let r: Vec<f32> = x.iter().map(|a|a.1.clone()).collect();
                        // let r = Array::from_vec(r);
                        // let mut a: Vec<(usize, f32)> = x
                        //     .iter()
                        //     .map(|x| (x.0.clone(), x.1.clone()))
                        //     .collect::<Vec<(usize, f32)>>();
                        // a.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        //debug!(x.clone());
                        return x;
                    }
                } else {
                    let t: &DecisionTreeClassifier<F>;
                    if x[node.split_idx.unwrap()] <= node.split_val.unwrap() {
                        t = &node.left;
                    } else {
                        t = &node.right;
                    }
                    return t.predict_row(x);
                }
            }

            _ => {
                panic!("shouldn't reach here");
            }
        }

        panic!("shouldn't reach here");
    }
}

impl<F: Float> DecisionTreeRegressor<F> {
    /// Constructor of DecisionTree
    fn new(
        idxs: Vec<usize>,
        size: usize,
        params: &RandomForrestParams,
        dataset: &DatasetBase<Array2<F>, Array1<F>>,
    ) -> Self {
        let mut tree = DecisionTreeRegressor(DecisionTree::NonEmpty(Box::new(DecisionTreeNode {
            idxs,
            size,
            value: None,
            score: F::infinity(),
            split_idx: None,
            split_val: None,
            left: DecisionTree::Empty,
            right: DecisionTree::Empty,
        })));

        tree.fit(dataset, params);
        tree
    }

    fn fit(&mut self, dataset: &DatasetBase<Array2<F>, Array1<F>>, params: &RandomForrestParams) {
        if let DecisionTree::NonEmpty(node) = &mut self.0 {
            // convert targets to one hot encoding
            let x = dataset.records().select(Axis(0), &node.idxs);
            let y = dataset.targets().select(Axis(0), &node.idxs);

            // y shape is (sample_size, label size)
            //let y = target_to_onehot::<L, F>(dataset.labels(), y.view());

            node.value = Some(NodeValue::RegressorNode(y.mean().unwrap()));
            //node.size = (dataset.records().nrows() as f64 * node.sample_pct) as usize;
            // println!("y: {:?}", y);
            // println!("value: {:?}", node.value);
            let c = y.shape()[1]; //node.num_features;
            for i in 0..c {
                // x shape is (sample_size, 1)
                let x = x.slice_axis(Axis(1), Slice::from(i..i + 1));
                self.find_best_split(x.view(), y.view(), i, params);
            }
            //fit left and right
        }
    }

    fn update_mse_split(
        &mut self,
        sort_x: ArrayView2<F>,
        sort_y: ArrayView1<F>,
        feature_idx: usize,
        params: &RandomForrestParams,
    ) {
        if let DecisionTree::NonEmpty(node) = &mut self.0 {
            let label_dim = sort_y.shape()[1];
            let mut r_cnt = F::from(node.size).unwrap();
            let mut r_sum = sort_y.sum();
            let mut r_sum2 = (&sort_y * &sort_y).sum();
            let (mut l_cnt, mut l_sum, mut l_sum2) = (
                F::zero(),
                // Array1::<F>::zeros(label_dim),
                // Array1::<F>::zeros(label_dim),
                F::zero(),
                F::zero(),
            );

            for i in 0..(node.size - params.min_leaf) {
                let xi = sort_x[[i, 0]];
                let yi = sort_y[i];
                let yi2 = yi * yi;
                l_cnt += F::one();
                r_cnt -= F::one();
                l_sum += yi;
                r_sum -= yi;
                l_sum2 += yi2;
                r_sum2 -= yi2;
                if i < params.min_leaf - 1 || xi == sort_x[[i + 1, 0]] {
                    continue;
                }
                let l_std = std_agg::<F>(l_cnt, l_sum, l_sum2);
                let r_std = std_agg::<F>(r_cnt, r_sum, r_sum2);

                let current_score = l_std * l_cnt + r_std * r_cnt;

                if current_score < node.score {
                    node.score = current_score;
                    node.split_idx = Some(feature_idx);
                    node.split_val = Some(xi);
                }
            }
        }
    }

    fn find_best_split(
        &mut self,
        // dataset: &DatasetBase<Array2<F>, Array2<L>>,
        x: ArrayView2<F>,
        y: ArrayView1<F>,
        feature_idx: usize,
        params: &RandomForrestParams,
    ) {
        if let DecisionTree::NonEmpty(node) = &self.0 {
            // sort by x's value and returns sorted idx list
            let mut sort_idx: Vec<(usize, _)> = x
                //.slice(s![.., feature_idx])
                .into_iter()
                .enumerate()
                .collect();
            sort_idx.sort_unstable_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            let sort_idx: Vec<usize> = sort_idx.into_iter().map(|a| a.0).collect();

            // sort_x shape (sample_size, 1)
            let sort_x = x.select(Axis(0), &sort_idx);
            // sort_y shape (sample_size, label size)
            let sort_y = y.select(Axis(0), &sort_idx);

            self.update_mse_split(sort_x.view(), sort_y.view(), feature_idx, params);
        }
    }
}

fn std_agg<F: Float>(cnt: F, sum: F, sum2: F) -> F {
    (sum2 / cnt) - (sum / cnt) * (sum / cnt)
    //res.mapv(|d| F::sqrt(d)).mean().unwrap()
}

fn gini_score<F: Float>(
    lhs: &mut Vec<HashMap<usize, usize>>,
    rhs: &mut Vec<HashMap<usize, usize>>,
    n_targets: usize,
    l_cnt: f64,
    r_cnt: f64,
) -> F {
    let mut score = 0.;
    for i in 0..n_targets {
        //lhs.len() {
        let li = &lhs[i];
        let ri = &rhs[i];
        // let lsz = li.iter().map(|x| x.1.clone() as f64).sum::<f64>();
        // let rsz = ri.iter().map(|x| x.1.clone() as f64).sum::<f64>();
        let sum_li = 1_f64
            - li.iter()
                .map(|x| (*x.1 as f64 / l_cnt as f64).powf(2.))
                .sum::<f64>();
        let sum_ri = 1_f64
            - ri.iter()
                .map(|x| (*x.1 as f64 / r_cnt as f64).powf(2.))
                .sum::<f64>();
        score += sum_li * l_cnt / (l_cnt + r_cnt) + sum_ri * r_cnt / (l_cnt + r_cnt);
    }
    F::from(score / n_targets as f64).unwrap()
}

/// convert target to one hot encoding
pub fn target_to_onehot<L: Label + Debug, F: Float>(labels: Vec<L>, y: ArrayView2<L>) -> Array2<F> {
    let mapping = labels
        .into_iter()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect::<HashMap<L, usize>>();
    println!("mapping: {:?}", mapping);
    let label_dim = mapping.len();
    let mut onehot = Array2::<F>::zeros((y.shape()[0], label_dim));

    // y shape (nrows, ntargets)
    for (i, row) in y.genrows().into_iter().enumerate() {
        for col in row.gencolumns() {
            let idx = mapping.get(&col[0]).unwrap();
            let idxs = s![i, *idx];
            onehot.slice_mut(idxs).fill(F::from_f64(1.0).unwrap());
        }
    }
    // onehot shape (nrows, label_dimension)
    onehot
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::error::Result;
    use ndarray::{s, Array};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};

    // #[test]
    // fn test_iris() -> Result<()> {
    //     let mut rng = Isaac64Rng::seed_from_u64(42);

    //     let (train, test) = linfa_datasets::iris()
    //         .shuffle(&mut rng)
    //         .split_with_ratio(0.8);

    //     println!("train record shape: {:?}", train.records().shape());
    //     println!("train target shape: {:?}", train.targets().shape());
    //     println!("train record sample: {:?}", train.records.slice(s![..2, ..]));
    //     println!("train target sample: {:?}", train.targets.slice(s![..2, ..]));

    //     let rf = RandomForrestClassifer::<f64, usize>::params()
    //         .num_trees(1)
    //         .sample_pct(1.)
    //         .min_leaf(1)
    //         .fit(&train);
    //     println!("tree: {:?}", rf.trees[0]);
    //     // let a = test.records();
    //     // let preds = rf.predict(a);
    //     //println!("predications: {:?}", preds);
    //     Ok(())
    // }

    // #[test]
    // fn single_feature_random_noise_binary() -> Result<()> {
    //     // generate data with 9 white noise and a single correlated feature
    //     let mut data = Array::random((50, 10), Uniform::new(-4., 4.));
    //     data.slice_mut(s![.., 0]).assign(
    //         &(0..50)
    //             .map(|x| if x < 2 { 0.0 } else { 1.0 })
    //             .collect::<Array1<_>>(),
    //     );

    //     let targets = (0..50).map(|x| x < 2).collect::<Array1<_>>();
    //     let dataset = Dataset::new(data, targets);
    //     //println!("dataset records {:#?}:", dataset.records()); //.slice(s![..2, ..]));
    //     //println!("dataset targets {:#?}:", dataset.targets()); //.slice(s![..2, ..]));
    //     let model = RandomForrestClassifer::<f64, usize>::params()
    //         .num_trees(1)
    //         .sample_pct(1.)
    //         .min_leaf(1)
    //         .fit(&dataset);

    //     //println!("model: {:?}", model.trees[0]);

    //     Ok(())
    // }
}
