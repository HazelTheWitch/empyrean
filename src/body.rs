pub mod generators;

use std::{
    fmt::{Debug, Display},
    num::NonZeroUsize,
};

use glam::DVec3;
use thiserror::Error;

use crate::error::{Validate, ValidationErrorAccumulator};

/// The basic data needed to create a [`Body`].
#[derive(Debug, Clone, Copy)]
pub struct BodyData {
    pub mass: f64,
    pub position: DVec3,
    pub velocity: DVec3,
}

impl BodyData {
    pub fn with_mass(mut self, mass: f64) -> Self {
        debug_assert!(mass.is_sign_positive() && mass.is_finite());

        self.mass = mass;
        self
    }

    pub fn with_position(mut self, position: DVec3) -> Self {
        debug_assert!(position.is_finite());

        self.position = position;
        self
    }

    pub fn with_velocity(mut self, velocity: DVec3) -> Self {
        debug_assert!(velocity.is_finite());

        self.velocity = velocity;
        self
    }
}

/// The id of a [`Body`] within a simulation.
///
/// Internally the id is stored as a [`NonZeroUsize`](std::num::NonZeroUsize) so `Option<BodyId>`
/// is the same size as `BodyId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(NonZeroUsize);

impl BodyId {
    const MASK: usize = usize::MAX;

    pub fn new(index: usize) -> Option<Self> {
        Some(Self(NonZeroUsize::new(index ^ Self::MASK)?))
    }

    pub fn index(&self) -> usize {
        usize::from(self.0) ^ Self::MASK
    }
}

impl Display for BodyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index())
    }
}

impl From<BodyId> for usize {
    fn from(BodyId(value): BodyId) -> Self {
        usize::from(value) ^ BodyId::MASK
    }
}

/// Container storing data about a body in a [`Simulation`](crate::simulation::Simulation)
/// as well as an id for tracking bodies through steps.
#[derive(Debug)]
pub struct Body {
    pub mass: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub(crate) id: BodyId,
}

impl Body {
    pub(crate) fn new(data: BodyData, index: usize) -> Option<Self> {
        let BodyData {
            mass,
            position,
            velocity,
        } = data;
        Some(Self {
            mass,
            position,
            velocity,
            id: BodyId::new(index)?,
        })
    }

    pub(crate) fn enforce_periodic_boundary(&mut self, half_size: f64) {
        self.position =
            (self.position + half_size).rem_euclid(DVec3::splat(2.0 * half_size)) - half_size
    }

    /// Retrieve the id of this body.
    pub fn id(&self) -> BodyId {
        self.id
    }

    /// Retrieve the data of this body, without the id.
    pub fn data(&self) -> BodyData {
        BodyData {
            mass: self.mass,
            position: self.position,
            velocity: self.velocity,
        }
    }
}

impl Validate for Body {
    fn accumulate_validation_errors(&self, errors: &mut ValidationErrorAccumulator) -> usize {
        let initial = errors.len();

        if !(self.mass.is_sign_positive() && self.mass.is_finite()) {
            errors.push(BodyValidationError::InvalidMass {
                id: self.id,
                mass: self.mass,
            });
        }

        if !self.position.is_finite() {
            errors.push(BodyValidationError::InvalidPosition {
                id: self.id,
                position: self.position,
            });
        }

        if !self.velocity.is_finite() {
            errors.push(BodyValidationError::InvalidVelocity {
                id: self.id,
                velocity: self.velocity,
            });
        }

        errors.len() - initial
    }
}

#[derive(Debug, Error)]
pub enum BodyValidationError {
    #[error("body {id} has invalid mass: '{mass}'")]
    InvalidMass { id: BodyId, mass: f64 },
    #[error("body {id} has invalid position: '{position}'")]
    InvalidPosition { id: BodyId, position: DVec3 },
    #[error("body {id} has invalid velocity: '{velocity}'")]
    InvalidVelocity { id: BodyId, velocity: DVec3 },
}
