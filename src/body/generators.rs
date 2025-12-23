use std::iter::repeat_with;

use glam::{DAffine3, DQuat, DVec2, DVec3};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{
    Distribution, Normal, Uniform, UnitBall, UnitDisc, uniform,
    weighted::{self, AliasableWeight, WeightedAliasIndex},
};

use crate::body::BodyData;

pub trait BodyGenerator {
    fn generate(&mut self) -> BodyData;

    fn iter(&mut self) -> impl Iterator<Item = BodyData> {
        repeat_with(move || self.generate())
    }
}

impl<G> BodyGenerator for Box<G>
where
    G: BodyGenerator,
{
    fn generate(&mut self) -> BodyData {
        self.as_mut().generate()
    }
}

impl<G> BodyGenerator for &mut G
where
    G: BodyGenerator,
{
    fn generate(&mut self) -> BodyData {
        (*self).generate()
    }
}

pub struct WeightedChoice<G, W: AliasableWeight, R> {
    choices: Vec<G>,
    distribution: WeightedAliasIndex<W>,
    rng: R,
}

impl<G, W, R> BodyGenerator for WeightedChoice<G, W, R>
where
    G: BodyGenerator,
    W: AliasableWeight,
    R: Rng,
{
    fn generate(&mut self) -> BodyData {
        self.choices[self.distribution.sample(&mut self.rng)].generate()
    }
}

#[derive(Clone, Copy)]
pub struct Modify<G, F> {
    generator: G,
    f: F,
}

impl<G, F> BodyGenerator for Modify<G, F>
where
    G: BodyGenerator,
    F: FnMut(&mut BodyData),
{
    fn generate(&mut self) -> BodyData {
        let mut data = self.generator.generate();
        (self.f)(&mut data);
        data
    }
}

#[derive(Clone, Copy)]
pub struct Map<G, F> {
    generator: G,
    f: F,
}

impl<G, F> BodyGenerator for Map<G, F>
where
    G: BodyGenerator,
    F: FnMut(BodyData) -> BodyData,
{
    fn generate(&mut self) -> BodyData {
        (self.f)(self.generator.generate())
    }
}

#[derive(Clone, Copy)]
pub struct FromFn<F> {
    f: F,
}

impl<F> BodyGenerator for FromFn<F>
where
    F: FnMut() -> BodyData,
{
    fn generate(&mut self) -> BodyData {
        (self.f)()
    }
}

#[derive(Clone, Copy)]
pub struct Transform<G> {
    generator: G,
    transformation: DAffine3,
}

impl<G> BodyGenerator for Transform<G>
where
    G: BodyGenerator,
{
    fn generate(&mut self) -> BodyData {
        let mut body = self.generator.generate();

        body.position = self.transformation.transform_point3(body.position);
        body.velocity = self.transformation.transform_vector3(body.velocity);

        body
    }
}

pub fn from_fn<F>(f: F) -> FromFn<F>
where
    F: FnMut() -> BodyData,
{
    FromFn { f }
}

/// Returns a [`BodyGenerator`] which places bodies on a disc at z = 0 with 0 velocity.
///
/// * `rng`: the random number generator to use
/// * `mass`: the mass of all the bodies
/// * `radius`: the radius of the disc in each dimension
///
/// ## See Also
///
/// [`BodyGeneratorExt::transform`] and [`BodyGeneratorExt::rotate`] for adjusting the orientation of the disc.
pub fn uniform_disc(mut rng: impl Rng, mass: f64, radius: DVec2) -> impl BodyGenerator {
    from_fn(move || BodyData {
        mass,
        position: (DVec2::from_array(UnitDisc.sample(&mut rng)) * radius).extend(0.0),
        velocity: DVec3::ZERO,
    })
}

/// Returns a [`BodyGenerator`] which places bodies in a cylinder with a given radius, height, 0 velocity.
///
/// * `rng`: the random number generator to use
/// * `mass`: the mass of all the bodies
/// * `radius`: the radius of the cylinder in each dimension
/// * `half_height`: half of the height of the cylinder, the maximum z value possible for bodies
///
/// ## See Also
///
/// [`BodyGeneratorExt::transform`] and [`BodyGeneratorExt::rotate`] for adjusting the orientation of the disc.
pub fn uniform_cylinder(
    mut rng: impl Rng,
    mass: f64,
    radius: DVec2,
    half_height: f64,
) -> Result<impl BodyGenerator, uniform::Error> {
    let distribution = Uniform::new_inclusive(-half_height, half_height)?;

    Ok(from_fn(move || BodyData {
        mass,
        position: (DVec2::from_array(UnitDisc.sample(&mut rng)) * radius)
            .extend(distribution.sample(&mut rng)),
        velocity: DVec3::ZERO,
    }))
}

/// Returns a [`BodyGenerator`] which places bodies in a ball with a given size, mass, and 0
/// velocity.
///
/// * `rng`: the random number generator to use
/// * `mass`: the mass of all the bodies
/// * `size`: the size of the ball in each dimension
pub fn uniform_ball(mut rng: impl Rng, mass: f64, size: DVec3) -> impl BodyGenerator {
    from_fn(move || BodyData {
        mass,
        position: DVec3::from_array(UnitBall.sample(&mut rng)) * size,
        velocity: DVec3::ZERO,
    })
}

/// Returns a [`BodyGenerator`] which places bodies with a set mass and 0 velocity according to a
/// 3d normal distribution centered at `(0, 0, 0)`.
///
/// * `rng`: the random number generator to use
/// * `mass`: the mass of all the bodies
/// * `position_standard_deviation`: the standard deviation in each dimension for the position of
///     the bodies
pub fn normal(
    mut rng: impl Rng,
    mass: f64,
    position_standard_deviation: DVec3,
) -> Result<impl BodyGenerator, rand_distr::NormalError> {
    let position = [
        Normal::new(0.0, position_standard_deviation.x)?,
        Normal::new(0.0, position_standard_deviation.y)?,
        Normal::new(0.0, position_standard_deviation.z)?,
    ];

    Ok(from_fn(move || BodyData {
        mass,
        position: DVec3::from_array(position.map(|d| d.sample(&mut rng))),
        velocity: DVec3::ZERO,
    }))
}

/// Returns a [`BodyGenerator`] which places bodies with a set mass according to a
/// 3d normal distribution centered at `(0, 0, 0)`.
///
/// Unlike [`normal`] this function also generates velocity based on a normal distribution.
///
/// * `rng`: the random number generator to use
/// * `mass`: the mass of all the bodies
/// * `position_standard_deviation`: the standard deviation in each dimension for the position of
///     the bodies
/// * `velocity_standard_deviation`: the standard deviation in each dimension for the velocity of
///     the bodies
pub fn normal_with_velocity(
    mut rng: impl Rng,
    mass: f64,
    position_standard_deviation: DVec3,
    velocity_standard_deviation: DVec3,
) -> Result<impl BodyGenerator, rand_distr::NormalError> {
    let velocity = [
        Normal::new(0.0, velocity_standard_deviation.x)?,
        Normal::new(0.0, velocity_standard_deviation.y)?,
        Normal::new(0.0, velocity_standard_deviation.z)?,
    ];

    Ok(normal(
        StdRng::from_rng(&mut rng),
        mass,
        position_standard_deviation,
    )?
    .modify(move |body| body.velocity = DVec3::from_array(velocity.map(|d| d.sample(&mut rng)))))
}

pub fn weighted_choice<G, W, R>(
    rng: R,
    choices: impl IntoIterator<Item = (W, G)>,
) -> Result<WeightedChoice<G, W, R>, weighted::Error>
where
    W: AliasableWeight,
    G: BodyGenerator,
    R: Rng,
{
    let (weights, choices) = choices.into_iter().unzip();

    let distribution = WeightedAliasIndex::new(weights)?;

    Ok(WeightedChoice {
        choices,
        distribution,
        rng,
    })
}

pub trait BodyGeneratorExt: BodyGenerator {
    fn boxed(self) -> impl BodyGenerator
    where
        Self: Sized,
    {
        Box::new(self)
    }

    fn map<F>(self, f: F) -> Map<Self, F>
    where
        F: FnMut(BodyData) -> BodyData,
        Self: Sized,
    {
        Map { generator: self, f }
    }

    fn modify<F>(self, f: F) -> Modify<Self, F>
    where
        F: FnMut(&mut BodyData),
        Self: Sized,
    {
        Modify { generator: self, f }
    }

    fn transform(self, transformation: DAffine3) -> Transform<Self>
    where
        Self: Sized,
    {
        Transform {
            generator: self,
            transformation,
        }
    }

    fn rotate(self, rotation: DQuat) -> Transform<Self>
    where
        Self: Sized,
    {
        Transform {
            generator: self,
            transformation: DAffine3::from_quat(rotation),
        }
    }

    fn translate(self, translation: DVec3) -> Transform<Self>
    where
        Self: Sized,
    {
        Transform {
            generator: self,
            transformation: DAffine3::from_translation(translation),
        }
    }

    fn scale(self, scale: DVec3) -> Transform<Self>
    where
        Self: Sized,
    {
        Transform {
            generator: self,
            transformation: DAffine3::from_scale(scale),
        }
    }

    fn with_mass(self, mut mass: impl FnMut() -> f64) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.map(move |data| data.with_mass(mass()))
    }

    fn with_position(self, mut position: impl FnMut() -> DVec3) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.map(move |data| data.with_position(position()))
    }

    fn with_velocity(self, mut velocity: impl FnMut() -> DVec3) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.map(move |data| data.with_velocity(velocity()))
    }

    fn with_mass_distribution(
        self,
        mass: impl Distribution<f64>,
        mut rng: impl Rng,
    ) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.with_mass(move || mass.sample(&mut rng))
    }

    fn with_position_component_distribution(
        self,
        position: impl Distribution<f64>,
        mut rng: impl Rng,
    ) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.with_position(move || {
            DVec3::new(
                position.sample(&mut rng),
                position.sample(&mut rng),
                position.sample(&mut rng),
            )
        })
    }

    fn with_velocity_component_distribution(
        self,
        velocity: impl Distribution<f64>,
        mut rng: impl Rng,
    ) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.with_position(move || {
            DVec3::new(
                velocity.sample(&mut rng),
                velocity.sample(&mut rng),
                velocity.sample(&mut rng),
            )
        })
    }

    fn with_position_distribution(
        self,
        position: impl Distribution<DVec3>,
        mut rng: impl Rng,
    ) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.with_position(move || position.sample(&mut rng))
    }

    fn with_velocity_distribution(
        self,
        velocity: impl Distribution<DVec3>,
        mut rng: impl Rng,
    ) -> impl BodyGenerator
    where
        Self: Sized,
    {
        self.with_velocity(move || velocity.sample(&mut rng))
    }
}

impl<G> BodyGeneratorExt for G where G: BodyGenerator {}
