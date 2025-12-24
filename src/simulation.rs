use std::iter::repeat_with;

use glam::DVec3;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    body::{Body, BodyData, generators::BodyGenerator},
    error::{Validate, ValidationErrorAccumulator},
    integrator::Integrator,
    octree::Octree,
};

#[derive(Debug, Error)]
pub enum SimulationValidationError {
    #[error("simulation has invalid time step: '{time_step}'")]
    InvalidTimeStep { time_step: f64 },
    #[error("simulation has invalid domain half-extent: '{half_size}'")]
    InvalidDomain { half_size: f64 },
}

pub struct Simulation<I: Integrator> {
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
    /// Scratch space used for storing integrator state for each body.
    states: Vec<I::State>,
    /// The integrator for approximating the solution to the equations of motion.
    integrator: I,
}

impl<I> Simulation<I>
where
    I: Integrator,
{
    /// Constructs a new [`Simulation`].
    ///
    /// * `theta`: see [`Octree::new`] for a description of this parameter
    /// * `epsilon`: see [`Octree::new`] for a description of this parameter
    /// * `time_step`: the time advanced in one step of the simulation
    /// * `half_size`: the half-extent of the domain of the simulation in every dimension
    pub const fn new(
        integrator: I,
        theta: f64,
        epsilon: f64,
        time_step: f64,
        half_size: f64,
    ) -> Self {
        Self {
            time_step,
            step: 0,
            half_size,
            bodies: Vec::new(),
            octree: Octree::new(theta, epsilon, half_size),
            states: Vec::new(),
            integrator,
        }
    }

    /// Advance the [`Simulation`] a single step.
    pub fn step(&mut self) {
        // Ensure terms has enough scratch space.
        if self.bodies.len() > self.states.len() {
            self.states.extend(
                repeat_with(|| I::State::default()).take(self.bodies.len() - self.states.len()),
            );
        }

        self.octree.build(&mut self.bodies);

        #[cfg(feature = "multi-thread")]
        {
            self.bodies
                .par_iter()
                .zip(self.states.par_iter_mut())
                .for_each(|(body, state)| {
                    let acceleration = |position| {
                        self.octree
                            .acceleration(position, &self.bodies, Some(body.id))
                    };

                    self.integrator
                        .update(self.time_step, body, state, acceleration);
                });

            self.bodies
                .par_iter_mut()
                .zip(self.states.par_iter())
                .for_each(|(body, state)| {
                    self.integrator.apply(
                        self.time_step,
                        state,
                        &mut body.position,
                        &mut body.velocity,
                    );

                    body.enforce_periodic_boundary(self.half_size);
                });
        }

        #[cfg(not(feature = "multi-thread"))]
        {
            for (body, state) in self.bodies.iter().zip(self.states.iter_mut()) {
                let acceleration = |position| {
                    self.octree
                        .acceleration(position, &self.bodies, Some(body.id))
                };

                self.integrator
                    .update(self.time_step, body, state, acceleration);
            }

            for (body, state) in self.bodies.iter_mut().zip(self.states.iter()) {
                self.integrator.apply(
                    self.time_step,
                    state,
                    &mut body.position,
                    &mut body.velocity,
                );

                body.enforce_periodic_boundary(self.half_size);
            }
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

impl<I> Validate for Simulation<I>
where
    I: Integrator,
{
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

impl<I> Extend<BodyData> for Simulation<I>
where
    I: Integrator,
{
    fn extend<T: IntoIterator<Item = BodyData>>(&mut self, bodies: T) {
        self.add_bodies(bodies);
    }
}
