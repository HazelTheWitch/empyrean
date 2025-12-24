use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use empyrean::{
    body::generators::{self, BodyGeneratorExt},
    integrator::{EulerMethod, Integrator, RungeKutta},
    simulation::Simulation,
};
use glam::DVec3;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Gamma};

const HALF_SIZE: f64 = 100.0;

fn new_simulation<I>(integrator: I, n: usize) -> Simulation<I>
where
    I: Integrator,
{
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

    let mut sim = Simulation::new(integrator, 1.0, 1.0, 1.0, HALF_SIZE);

    sim.generate_bodies(&mut generator, n);

    sim
}

fn simulation_benchmark_runge_kutta(c: &mut Criterion) {
    let mut runge_kutta_group = c.benchmark_group("simulation step - runge-kutta");

    for body_count in (1..=4).map(|p| 10usize.pow(p)) {
        runge_kutta_group.bench_with_input(
            BenchmarkId::from_parameter(body_count),
            &body_count,
            |b, body_count| {
                b.iter_batched(
                    || new_simulation(RungeKutta, *body_count),
                    |mut sim| sim.step(),
                    BatchSize::LargeInput,
                )
            },
        );
    }
}

fn simulation_benchmark_euler_method(c: &mut Criterion) {
    let mut euler_method_group = c.benchmark_group("simulation step - euler method");

    for body_count in (1..=4).map(|p| 10usize.pow(p)) {
        euler_method_group.bench_with_input(
            BenchmarkId::from_parameter(body_count),
            &body_count,
            |b, body_count| {
                b.iter_batched(
                    || new_simulation(EulerMethod, *body_count),
                    |mut sim| sim.step(),
                    BatchSize::LargeInput,
                )
            },
        );
    }
}

criterion_group!(
    benches,
    simulation_benchmark_runge_kutta,
    simulation_benchmark_euler_method
);
criterion_main!(benches);
