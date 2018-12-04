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

struct TeachingData {
    input: (i8, i8, i8),
    expected: i8,
}

extern crate rand;

use rand::Rng;
use rand::distributions::{Range, IndependentSample};

fn main() {
    let mut rng = rand::thread_rng();

    // Teaching data provided as an array of TeachingData instances

    // Implementaion of a 2-input OR gate
    // let teaching_data = [
    //     TeachingData { input: (0, 0, 1), expected: 0 },
    //     TeachingData { input: (0, 1, 1), expected: 1 },
    //     TeachingData { input: (1, 0, 1), expected: 1 },
    //     TeachingData { input: (1, 1, 1), expected: 1 },
    // ];

    // Implementaion of a 2-input AND gate
    // let teaching_data = [
    //     TeachingData { input: (0, 0, 1), expected: 0 },
    //     TeachingData { input: (0, 1, 1), expected: 0 },
    //     TeachingData { input: (1, 0, 1), expected: 0 },
    //     TeachingData { input: (1, 1, 1), expected: 1 },
    // ];

    // Implementaion of a 2-input NOR gate
    // let teaching_data = [
    //     TeachingData { input: (0, 0, 1), expected: 1 },
    //     TeachingData { input: (0, 1, 1), expected: 0 },
    //     TeachingData { input: (1, 0, 1), expected: 0 },
    //     TeachingData { input: (1, 1, 1), expected: 0 },
    // ];


    // Implementaion of a 2-input NAND gate
    let teaching_data = [
        TeachingData { input: (0, 0, 1), expected: 1 },
        TeachingData { input: (0, 1, 1), expected: 1 },
        TeachingData { input: (1, 0, 1), expected: 1 },
        TeachingData { input: (1, 1, 1), expected: 0 },
    ];


    // // Implementation of a 2-input EXOR gate
    // let teaching_data = [
    //     TeachingData { input: (0, 0, 1), expected: 0},
    //     TeachingData { input: (0, 1, 1), expected: 1},
    //     TeachingData { input: (1, 0, 1), expected: 1},
    //     TeachingData { input: (1, 1, 1), expected: 0},
    // ];


    // Initialize the weight vector with random data between 0 and 1
    let range = Range::new(0.0, 1.0);
    let mut w = (
        range.ind_sample(&mut rng),
        range.ind_sample(&mut rng),
        range.ind_sample(&mut rng),
    );

    // Learning rate is set to 0.2 along with the iteration count to 80
    let eta = 0.2;
    let n = 10000;

    // Teaching
    println!("Starting teaching phase with {} iterations.", n);
    for _ in 0..n {

        // Choose a random teaching sample
        let &TeachingData { input: x, expected } = rng.choose(&teaching_data).unwrap();

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
    for &TeachingData { input, .. } in &teaching_data {
        let result = dot(input, w);
        println!("({} :: {}): {:.*} -> {}", input.0, input.1, 8, result, result.heaviside());
    }
}
