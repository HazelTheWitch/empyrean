use thiserror::Error;

use crate::{body::BodyValidationError, simulation::SimulationValidationError};

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    ValidationError(#[from] ValidationError),
}

pub trait Validate {
    /// Validates this object's state, returning the number of errors found.
    ///
    /// * `errors`: the list of errors to accumulate into
    fn accumulate_validation_errors(&self, errors: &mut ValidationErrorAccumulator) -> usize;

    fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errors = ValidationErrorAccumulator::default();

        self.accumulate_validation_errors(&mut errors);

        errors.into_result()
    }
}

#[derive(Debug, Default)]
pub struct ValidationErrorAccumulator(Vec<ValidationError>);

impl ValidationErrorAccumulator {
    pub fn push(&mut self, error: impl Into<ValidationError>) {
        self.0.push(error.into());
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_result(self) -> Result<(), Vec<ValidationError>> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(self.into_inner())
        }
    }

    pub fn into_inner(self) -> Vec<ValidationError> {
        self.0
    }
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error(transparent)]
    Simulation(#[from] SimulationValidationError),
    #[error(transparent)]
    Body(#[from] BodyValidationError),
}
