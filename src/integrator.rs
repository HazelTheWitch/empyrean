use glam::DVec3;

use crate::body::Body;

pub trait Integrator {
    type State: Default;

    fn apply(
        &self,
        time_step: f64,
        state: &Self::State,
        position: &mut DVec3,
        velocity: &mut DVec3,
    );
    fn update(
        &self,
        time_step: f64,
        body: &Body,
        state: &mut Self::State,
        acceleration: impl Fn(DVec3) -> DVec3,
    );
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EulerMethod;

impl Integrator for EulerMethod {
    type State = DVec3;

    fn apply(
        &self,
        time_step: f64,
        acceleration: &Self::State,
        position: &mut DVec3,
        velocity: &mut DVec3,
    ) {
        *velocity += *acceleration * time_step;
        *position += *velocity * time_step;
    }

    fn update(
        &self,
        _: f64,
        body: &Body,
        state: &mut Self::State,
        acceleration: impl Fn(DVec3) -> DVec3,
    ) {
        *state = acceleration(body.position);
    }
}

/// 4th order Runge-Kutta integrator
#[derive(Debug, Default, Clone, Copy)]
pub struct RungeKutta;

impl Integrator for RungeKutta {
    type State = RungeKuttaTerms;

    #[inline]
    fn apply(
        &self,
        time_step: f64,
        state: &Self::State,
        position: &mut DVec3,
        velocity: &mut DVec3,
    ) {
        *velocity += state.delta_velocity(time_step);
        *position += state.delta_position(time_step);
    }

    #[inline]
    fn update(
        &self,
        time_step: f64,
        body: &Body,
        state: &mut Self::State,
        acceleration: impl Fn(DVec3) -> DVec3,
    ) {
        let half_step = time_step / 2.0;

        state.k1_velocity = acceleration(body.position);
        state.k1_position = body.velocity;

        state.k2_velocity = acceleration(body.position + half_step * state.k1_position);
        state.k2_position = body.velocity + half_step * state.k1_velocity;

        state.k3_velocity = acceleration(body.position + half_step * state.k2_position);
        state.k3_position = body.velocity + half_step * state.k2_velocity;

        state.k4_velocity = acceleration(body.position + time_step * state.k3_position);
        state.k4_position = body.velocity + time_step * state.k3_velocity;
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RungeKuttaTerms {
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
