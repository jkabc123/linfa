use std::fs::File;
use std::io::Write;

use ndarray_rand::rand::SeedableRng;


use linfa::prelude::*;

fn main() -> Result<()> {
  // load Iris dataset
  let mut rng = SmallRng::seed_from_u64(42);

  let (train, test) = linfa_datasets::iris()
      .shuffle(&mut rng)
      .split_with_ratio(0.8);
}
