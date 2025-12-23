use glam::DVec3;

/// Partitions a slice based on a predicate.
///
/// Places all elements for which `predicate` is true are positioned before all elements for which
/// it is false. Then returns the index of the first element which returns false.
///
/// * `data`: the slice to partition
/// * `predicate`: the predicate to partition on
pub fn partition<T, F: Fn(&T) -> bool>(data: &mut [T], predicate: F) -> usize {
    if data.is_empty() {
        return 0;
    }

    let mut left = 0;
    let mut right = data.len() - 1;

    loop {
        while let Some(item) = data.get(left)
            && predicate(item)
        {
            left += 1;
        }

        while let Some(item) = data.get(right)
            && !predicate(item)
            && right > 0
        {
            right -= 1;
        }

        if left >= right {
            return left;
        }

        data.swap(left, right);
    }
}

/// Computes the distance direction vector between two points.
///
/// Assumes both points are within the simulation space.
///
/// * `from`: the first point
/// * `to`: the second point
/// * `half_size`: the half-extent of the simulation space
///
/// ## See Also
///
/// https://en.wikipedia.org/wiki/Periodic_boundary_conditions#(A)_Restrict_particle_coordinates_to_the_simulation_box
pub fn periodic_boundary_distance_direction(from: DVec3, to: DVec3, half_size: f64) -> DVec3 {
    let direction = to - from;

    let half_size = DVec3::splat(half_size);
    let unoptimal = direction.abs().cmpgt(half_size);

    direction - 2.0 * (DVec3::from(unoptimal) * half_size.copysign(direction))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use glam::{IVec3, dvec3, ivec3};

    use crate::utils::{partition, periodic_boundary_distance_direction};

    #[test]
    fn test_periodic_boundary_distance() {
        let a = dvec3(-2.0, -1.0, -3.0);
        let b = dvec3(2.0, 1.0, 3.0);

        let d_ab = periodic_boundary_distance_direction(a, b, 3.0);
        let d_ba = periodic_boundary_distance_direction(b, a, 3.0);

        assert_relative_eq!(d_ab, -d_ba);

        assert_relative_eq!(d_ab, dvec3(-2.0, 2.0, 0.0));
    }

    #[test]
    fn test_partition() {
        const N: i32 = 5;

        let mut values: Vec<_> = (-N..N)
            .flat_map(|x| (-N..N).flat_map(move |y| (-N..N).map(move |z| ivec3(x, y, z))))
            .collect();

        let predicates: Vec<Box<dyn Fn(&IVec3) -> bool>> = vec![
            Box::new(|_| true),
            Box::new(|p| p.x > 3),
            Box::new(|p| p.x > 3 && p.z < -2),
            Box::new(|p| p.x > 3 && p.z < -2 && p.y % 3 == 1),
            Box::new(|_| false),
        ];

        for (index, predicate) in predicates.into_iter().enumerate() {
            let first_false = partition(&mut values, &predicate);

            for i in 0..first_false {
                assert!(
                    predicate(&values[i]),
                    "predicate = predicates[{index}] expected = true i = {i:<4} first_false = {first_false:<4} value = {}",
                    values[i]
                );
            }

            for i in first_false..values.len() {
                assert!(
                    !predicate(&values[i]),
                    "predicate = predicates[{index}] expected = false i = {i:<4} first_false = {first_false:<4} value = {}",
                    values[i]
                );
            }
        }
    }
}
