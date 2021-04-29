use linfa_randomforrest::{RandomForrestClassifer, debug};

use linfa::error::Result;
use linfa::metrics::ToConfusionMatrix;
use linfa::prelude::*;
use ndarray::{array, s, Array};
use ndarray_rand::rand::SeedableRng;
use rand_isaac::Isaac64Rng;
//use serde_json::{Result, Value};
use std::fs;
use std::io::BufReader;



#[test]
fn test_iris() -> Result<()> {
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("train record shape: {:?}", train.records().shape());
    println!("train target shape: {:?}", train.targets().shape());
    println!(
        "train record sample: {:?}",
        train.records.slice(s![..2, ..])
    );
    println!(
        "train target sample: {:?}",
        train.targets.slice(s![..2, ..])
    );

    let rf = RandomForrestClassifer::<f64, usize>::params()
        .num_trees(100)
        .sample_pct(0.1)
        .min_leaf(5)
        .fit(&train);
    //println!("tree: {:?}", rf.trees[0]);
    //let a = test.records();
    let ground_truth = test.targets.clone();
    let ground_truth = ground_truth.slice(s![.., 0]);
    let prediction = rf.predict(test);

    // println!("predications: {:?}", prediction.targets.shape());
    // println!("truth: {:?}", ground_truth.shape());
    //let cm = prediction.confusion_matrix(&ground_truth);

    // create dummy classes 0 and 1
    // let prediction1 =   array![0, 1, 1, 1, 0, 0, 1];
    // let ground_truth1 = array![0, 0, 1, 0, 1, 0, 1];
    // let cm1 = prediction1.confusion_matrix(ground_truth1).unwrap();
    // println!("test cm: {:?}", cm1);

    //columns are ground truth, rows are predictions
    //let prediction = prediction.targets.clone();
    //println!("predictions: {:?}", prediction);
    let cm = prediction.confusion_matrix(ground_truth).unwrap();
    println!("matrix: {:?}", cm);
    println!("precision: {:?}", cm.precision());

    // let rf_json = serde_json::to_string(&rf).unwrap();
    // debug!(rf_json);
    Ok(())
}


#[test]
fn test_serde() -> std::io::Result<()>{
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let (train, test) = linfa_datasets::iris()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    let rf = RandomForrestClassifer::<f64, usize>::params()
        .num_trees(100)
        .sample_pct(0.1)
        .min_leaf(5)
        .fit(&train);

    // let writer = fs::File::create("tmp.model")?;
    // serde_json::to_writer(writer, &rf);
    
    // let file = fs::File::open("tmp.model")?;
    // let reader = BufReader::new(file);
    
    //let rf_new = serde_json::from_reader(reader)?;
    //debug!(rf_new);
    // let ground_truth = test.targets.clone();
    // let ground_truth = ground_truth.slice(s![.., 0]);
    // let prediction = rf_new.predict(test);

    // let cm = prediction.confusion_matrix(ground_truth).unwrap();
    // println!("matrix: {:?}", cm);
    // println!("precision: {:?}", cm.precision());

    let serialized = serde_json::to_string(&rf).unwrap();
    println!("serialized = {}", serialized);
    let deserialized:  RandomForrestClassifer::<f64, usize> = serde_json::from_str(&serialized).unwrap();
    println!("deserialized = {:?}", deserialized);

    //fs::remove_file("tmp.model")?;
    Ok(())

   
 
}
