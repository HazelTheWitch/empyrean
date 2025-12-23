use std::iter::repeat_n;

use glam::DVec3;
use thiserror::Error;

use crate::{
    body::{Body, BodyData, generators::BodyGenerator},
    error::{Validate, ValidationErrorAccumulator},
    octree::Octree,
};

#[derive(Debug, Error)]
pub enum SimulationValidationError {
    #[error("simulation has invalid time step: '{time_step}'")]
    InvalidTimeStep { time_step: f64 },
    #[error("simulation has invalid domain half-extent: '{half_size}'")]
    InvalidDomain { half_size: f64 },
}

pub struct Simulation {
    /// The time advanced with each step.
    time_step: f64,
    /// The current step of the simulation, incremented with each call to [`Simulation::step`].
    step: u64,
    /// The half-extent of the domain of the simulation in every dimension.
    half_size: f64,
    /// The set of all bodies in this simulation.
    ///
    /// For the id within the [`bodies`](Body) to remain valid, no bodies can be removed from a
    /// simulation.
    bodies: Vec<Body>,
    /// The octree used for estimation of gravity between the bodies.
    octree: Octree,
    /// Scratch space used for computing RK4 terms for each body.
    terms: Vec<RungeKuttaTerms>,
}

impl Simulation {
    /// Constructs a new [`Simulation`].
    ///
    /// * `theta`: see [`Octree::new`] for a description of this parameter
    /// * `epsilon`: see [`Octree::new`] for a description of this parameter
    /// * `time_step`: the time advanced in one step of the simulation
    /// * `half_size`: the half-extent of the domain of the simulation in every dimension
    pub const fn new(theta: f64, epsilon: f64, time_step: f64, half_size: f64) -> Self {
        assert!(half_size > 0.0);

        Self {
            time_step,
            step: 0,
            half_size,
            bodies: Vec::new(),
            octree: Octree::new(theta, epsilon, half_size),
            terms: Vec::new(),
        }
    }

    /// Advance the [`Simulation`] a single step.
    pub fn step(&mut self) {
        // Ensure terms has enough scratch space.
        if self.bodies.len() > self.terms.len() {
            self.terms.extend(repeat_n(
                RungeKuttaTerms::default(),
                self.bodies.len() - self.terms.len(),
            ));
        }

        self.octree.build(&mut self.bodies);

        let half_step = self.time_step / 2.0;

        for (body, terms) in self.bodies.iter().zip(self.terms.iter_mut()) {
            terms.k1_velocity =
                self.octree
                    .acceleration(body.position, &self.bodies, Some(body.id));
            terms.k1_position = body.velocity;

            terms.k2_velocity = self.octree.acceleration(
                body.position + half_step * terms.k1_position,
                &self.bodies,
                Some(body.id),
            );
            terms.k2_position = body.velocity + half_step * terms.k1_velocity;

            terms.k3_velocity = self.octree.acceleration(
                body.position + half_step * terms.k2_position,
                &self.bodies,
                Some(body.id),
            );
            terms.k3_position = body.velocity + half_step * terms.k2_velocity;

            terms.k4_velocity = self.octree.acceleration(
                body.position + self.time_step * terms.k3_position,
                &self.bodies,
                Some(body.id),
            );
            terms.k4_position = body.velocity + self.time_step * terms.k3_velocity;
        }

        for (body, terms) in self.bodies.iter_mut().zip(self.terms.iter()) {
            body.velocity += terms.delta_velocity(self.time_step);
            body.position += terms.delta_position(self.time_step);

            body.enforce_periodic_boundary(self.half_size);
        }

        self.step += 1;
    }

    /// Get the current step index.
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Get the current time since the simulation began.
    pub fn current_time(&self) -> f64 {
        self.time_step * self.step as f64
    }

    /// Gets a slice of the bodies within this simulation.
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// Gets a mutable slice of the bodies within this simulation.
    pub fn bodies_mut(&mut self) -> &mut [Body] {
        &mut self.bodies
    }

    /// Copies the data from each body into a vector.
    ///
    /// The index of each element of the vector correcsponds to the id of the body it came from.
    ///
    /// ## See Also
    ///
    /// * [`Body::id`]
    #[must_use]
    pub fn body_data(&self) -> Vec<BodyData> {
        let mut data = vec![
            BodyData {
                mass: 0.0,
                position: DVec3::ZERO,
                velocity: DVec3::ZERO
            };
            self.bodies.len()
        ];

        for body in &self.bodies {
            data[body.id.index()] = body.data();
        }

        data
    }

    /// Generate bodies from a [`BodyGenerator`] and add them to the simulation.
    ///
    /// * `generator`: the [`BodyGenerator`] to pull from
    /// * `n`: the number of bodies to generate
    pub fn generate_bodies(&mut self, generator: &mut impl BodyGenerator, n: usize) {
        self.add_bodies(generator.iter().take(n));
    }

    /// Add a body to the simulation.
    ///
    /// * `body`: the data of the body to add to the simulation
    pub fn add_body(&mut self, body: impl Into<BodyData>) {
        let mut body = Body::new(body.into(), self.bodies.len()).expect("bodies full");

        body.enforce_periodic_boundary(self.half_size);

        self.bodies.push(body);
    }

    /// Extend bodies from an iterator.
    ///
    /// * `bodies`: an iterator of data to add to the simulation
    pub fn add_bodies(&mut self, bodies: impl IntoIterator<Item = impl Into<BodyData>>) {
        let first_id = self.bodies.len();

        self.bodies
            .extend(bodies.into_iter().enumerate().map(|(delta_id, data)| {
                let mut body = Body::new(data.into(), first_id + delta_id).expect("bodies full");

                body.enforce_periodic_boundary(self.half_size);

                body
            }));
    }
}

impl Validate for Simulation {
    fn accumulate_validation_errors(&self, errors: &mut ValidationErrorAccumulator) -> usize {
        let initial = errors.len();

        if !(self.time_step.is_sign_positive() && self.time_step.is_finite()) {
            errors.push(SimulationValidationError::InvalidTimeStep {
                time_step: self.time_step,
            });
        }

        if !(self.half_size.is_sign_positive() && self.half_size.is_finite()) {
            errors.push(SimulationValidationError::InvalidDomain {
                half_size: self.half_size,
            });
        }

        for body in &self.bodies {
            body.accumulate_validation_errors(errors);
        }

        errors.len() - initial
    }
}

impl Extend<BodyData> for Simulation {
    fn extend<T: IntoIterator<Item = BodyData>>(&mut self, bodies: T) {
        self.add_bodies(bodies);
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct RungeKuttaTerms {
    pub k1_position: DVec3,
    pub k1_velocity: DVec3,
    pub k2_position: DVec3,
    pub k2_velocity: DVec3,
    pub k3_position: DVec3,
    pub k3_velocity: DVec3,
    pub k4_position: DVec3,
    pub k4_velocity: DVec3,
}

impl RungeKuttaTerms {
    pub fn delta_position(&self, delta_time: f64) -> DVec3 {
        (delta_time / 6.0)
            * (self.k1_position + 2.0 * (self.k2_position + self.k3_position) + self.k4_position)
    }

    pub fn delta_velocity(&self, delta_time: f64) -> DVec3 {
        (delta_time / 6.0)
            * (self.k1_velocity + 2.0 * (self.k2_velocity + self.k3_velocity) + self.k4_velocity)
    }
}
