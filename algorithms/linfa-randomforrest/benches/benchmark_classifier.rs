#![feature(test)]

extern crate test;
use test::Bencher;

use linfa_randomforrest::RandomForrestClassifer;

use linfa::prelude::*;
use ndarray::s;
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;

#[bench]
fn bench_iris_classifier(b: &mut Bencher) {
    b.iter(|| {
        let mut rng = Isaac64Rng::seed_from_u64(42);

        let (train, test) = linfa_datasets::iris()
            .shuffle(&mut rng)
            .split_with_ratio(0.8);

        let rf = RandomForrestClassifer::<f64, usize>::params()
            .num_trees(100)
            .sample_pct(0.1)
            .min_leaf(5)
            .fit(&train);

        // let ground_truth = test.targets.clone();
        // let ground_truth = ground_truth.slice(s![.., 0]);
        // let prediction = rf.predict(test);

        // let prediction = prediction.targets.clone();
        // let cm = prediction.confusion_matrix(ground_truth).unwrap();
        // println!("matrix: {:?}", cm);
        // println!("precision: {:?}", cm.precision());
    });
}
