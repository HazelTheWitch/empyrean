use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use empyrean::{
    body::generators::{self, BodyGenerator, BodyGeneratorExt},
    simulation::Simulation,
};
use glam::DVec3;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Gamma};

const HALF_SIZE: f64 = 100.0;

fn simulation_with_bodies(
    theta: f64,
    epsilon: f64,
    time_step: f64,
    generator: &mut impl BodyGenerator,
    n: usize,
) -> Simulation {
    let mut sim = Simulation::new(theta, epsilon, time_step, HALF_SIZE);

    sim.generate_bodies(generator, n);

    sim
}

fn simulation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation step - normal bodies");

    let mut rng = rand::rng();

    let mass = Gamma::new(2.0, 1.0).unwrap();

    let mut generator = generators::normal_with_velocity(
        StdRng::from_rng(&mut rng),
        1.0,
        DVec3::splat(30.0),
        DVec3::splat(15.0),
    )
    .unwrap()
    .modify(|body| body.mass = mass.sample(&mut rng));

    for body_count in (1..=4).map(|p| 10usize.pow(p)) {
        group.bench_with_input(
            BenchmarkId::from_parameter(body_count),
            &body_count,
            |b, body_count| {
                b.iter_batched(
                    || simulation_with_bodies(1.0, 1.0, 1.0, &mut generator, *body_count),
                    |mut sim| sim.step(),
                    BatchSize::LargeInput,
                )
            },
        );
    }
}

criterion_group!(benches, simulation_benchmark);
criterion_main!(benches);
