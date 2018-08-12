// Heaviside step function
trait Heaviside {
    fn heaviside(&self) -> i8;
}

// Implement heaviside() for f64
impl Heaviside for f64 {
    fn heaviside(&self) -> i8 {
        (*self >= 0.0) as i8
    }
}

// Dot product of input and weights
fn dot(input: (i8, i8, i8), weights: (f64, f64, f64)) -> f64 {
    input.0 as f64 * weights.0
    + input.1 as f64 * weights.1
    + input.2 as f64 * weights.2
}

struct TrainingData {
    input: (i8, i8, i8),
    expected: i8,
}

extern crate rand;

use rand::Rng;
use rand::distributions::{Range, IndependentSample};

fn main() {
    let mut rng = rand::thread_rng();

    // Training data provided as an array of TrainingData instances

    //Implementaion of OR function
    // let training_data = [
    //     TrainingData { input: (0, 0, 1), expected: 0 },
    //     TrainingData { input: (0, 1, 1), expected: 1 },
    //     TrainingData { input: (1, 0, 1), expected: 1 },
    //     TrainingData { input: (1, 1, 1), expected: 1 },
    // ];

    //Implementaion of AND function
    // let training_data = [
    //     TrainingData { input: (0, 0, 1), expected: 0 },
    //     TrainingData { input: (0, 1, 1), expected: 0 },
    //     TrainingData { input: (1, 0, 1), expected: 0 },
    //     TrainingData { input: (1, 1, 1), expected: 1 },
    // ];

    //Implementaion of NOR function
    // let training_data = [
    //     TrainingData { input: (0, 0, 1), expected: 1 },
    //     TrainingData { input: (0, 1, 1), expected: 0 },
    //     TrainingData { input: (1, 0, 1), expected: 0 },
    //     TrainingData { input: (1, 1, 1), expected: 0 },
    // ];


    //Implementaion of NAND function
    let training_data = [
        TrainingData { input: (0, 0, 1), expected: 1 },
        TrainingData { input: (0, 1, 1), expected: 1 },
        TrainingData { input: (1, 0, 1), expected: 1 },
        TrainingData { input: (1, 1, 1), expected: 0 },
    ];


    // Initialize the weight vector with random data between 0 and 1
    let range = Range::new(0.0, 1.0);
    let mut w = (
        range.ind_sample(&mut rng),
        range.ind_sample(&mut rng),
        range.ind_sample(&mut rng),
    );

    // Learning rate is set to 0.2 along with the iteration count to 80
    let eta = 0.2;
    let n = 100;

    // Training
    println!("Starting training phase with {} iterations.\nTraining completed!", n);
    for _ in 0..n {

        // Choose a random training sample
        let &TrainingData { input: x, expected } = rng.choose(&training_data).unwrap();

        // Calculate the dot product
        let result = dot(x, w);

        // Calculate the error
        let error = expected - result.heaviside();
        // println!("{} {}: {} --> {} \n({}, {}, {})", x.0, x.1, result, error, w.0, w.1, w.2);

        // Update the weights
        w.0 += eta * error as f64 * x.0 as f64;
        w.1 += eta * error as f64 * x.1 as f64;
        w.2 += eta * error as f64 * x.2 as f64;
    }

    // After n iterations our perceptron should have learned by now.
    for &TrainingData { input, .. } in &training_data {
        let result = dot(input, w);
        println!("({} :: {}): {:.*} -> {}", input.0, input.1, 8, result, result.heaviside());
    }
}
