use std::fs::File;
use std::io::Write;

use linfa::dataset::Labels;
use ndarray::{array, s};
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use linfa::prelude::*;

fn main() -> Result<()> {
    // load Iris dataset
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("train nsamples: {}", train.nsamples());
    let indexes = s![..5, ..];
    println!("train targets: {:#?}", train.targets().slice(indexes));
    println!("train records: {:#?}", train.records().slice(indexes));
    println!("feature names: {:#?}", train.feature_names());
    println!("num of targets: {:#?}", train.ntargets());
    println!("label frequencies: {:#?}", train.label_frequencies());
    println!("labels: {:#?}", train.labels());

    //   // ------ Labels ------
    //   let dataset_multiclass = Dataset::from((
    //     array![[1., 2.], [2., 1.], [0., 0.], [2., 2.]],
    //     array![0, 1, 2, 2],
    // ));
    // let datasets_one_vs_all = train.one_vs_all().unwrap();
    // for dataset in datasets_one_vs_all.iter() {
    //   println!("labels:{:?}",dataset.labels());
    // }

    Ok(())
}
