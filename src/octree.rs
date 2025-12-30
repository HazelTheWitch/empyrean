#[cfg(feature = "multi-thread")]
use std::sync::atomic::AtomicUsize;
use std::{num::NonZeroUsize, ops::Range};

use glam::{DVec3, dvec3};
#[cfg(feature = "multi-thread")]
use rayon::Scope;

use crate::{
    body::{Body, BodyId},
    utils::{partition, periodic_boundary_distance_direction},
};

const LEAF_CAPACITY: usize = 64;

#[derive(Debug, Clone, Copy)]
struct Octant {
    pub center: DVec3,
    pub half_size: f64,
}

impl Octant {
    #[rustfmt::skip]
    fn subdivide(self) -> [Self; 8] {
        let half_size = self.half_size / 2.0;

        [
            Self { center: self.center + dvec3(-half_size, -half_size, -half_size), half_size },
            Self { center: self.center + dvec3( half_size, -half_size, -half_size), half_size },
            Self { center: self.center + dvec3(-half_size,  half_size, -half_size), half_size },
            Self { center: self.center + dvec3( half_size,  half_size, -half_size), half_size },
            Self { center: self.center + dvec3(-half_size, -half_size,  half_size), half_size },
            Self { center: self.center + dvec3( half_size, -half_size,  half_size), half_size },
            Self { center: self.center + dvec3(-half_size,  half_size,  half_size), half_size },
            Self { center: self.center + dvec3( half_size,  half_size,  half_size), half_size },
        ]
    }
}

type NonRootNodeIndex = NonZeroUsize;

#[derive(Debug, Clone)]
struct Node {
    /// A reference to the this node's first child.
    pub children: Option<NonRootNodeIndex>,
    /// The "next" node's index.
    ///
    /// For octants 0..7 this refers to the next octant, for octant 7 (-x, -y, -z) it is the parent's next.
    /// The root node does not have a next node.
    pub next: Option<NonRootNodeIndex>,
    /// The total mass of all bodies within this node.
    pub mass: f64,
    /// The center of mass of all bodies within this node.
    pub center_of_mass: DVec3,
    /// The octant this node covers.
    pub octant: Octant,
    /// The range of bodies this node contains.
    pub bodies: Range<usize>,
}

impl Node {
    pub const fn empty() -> Self {
        Self::new(
            None,
            Octant {
                center: DVec3::ZERO,
                half_size: 0.0,
            },
            0..0,
        )
    }

    pub const fn new(next: Option<NonRootNodeIndex>, octant: Octant, bodies: Range<usize>) -> Self {
        Self {
            children: None,
            next,
            mass: 0.0,
            center_of_mass: DVec3::ZERO,
            octant,
            bodies,
        }
    }

    pub fn children_range(&self) -> Option<Range<usize>> {
        let index = self.children?.into();

        Some(index..(index + 8))
    }
}

/// An octree implementing the [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
/// for simulating the n-body gravity simulation.
///
/// ## Octant Indices
///
/// Octants are indexed 0..8 based on their relative position. The bits of this index are based on
/// the which side the octants are on for each coordinate in the order ZYX, the bit is set if the
/// coordinate is less than the center of the parent.
///
/// ▲ +Z Octants           ▲ -Z Octants
/// │ ┌───────┬───────┐    │ ┌───────┬───────┐
/// │ │  001  │  000  │    │ │  101  │  100  │
/// │ │   1   │   0   │    │ │   5   │   4   │
/// │ │       │       │    │ │       │       │
/// Y ├───────┼───────┤    Y ├───────┼───────┤
/// │ │  011  │  010  │    │ │  111  │  110  │
/// │ │   3   │   2   │    │ │   7   │   6   │
/// │ │       │       │    │ │       │       │
/// │ └───────┴───────┘    │ └───────┴───────┘
/// └─────────X────────▶   └─────────X────────▶
#[derive(Debug)]
pub struct Octree {
    /// All nodes within the [`Octree`].
    ///
    /// Nodes are ordered such that parents are before children, and the root node is always at
    /// index 0.
    nodes: Vec<Node>,
    /// Contains the indices of nodes which are parents. In other words, all the nodes which are
    /// not leaves.
    parents: Vec<usize>,
    /// Theta cutoff for using [`Octree`] nodes to approximate gravitational forces.
    theta_squared: f64,
    /// Softening parameter
    epsilon_squared: f64,
    /// The half-extent of the root node.
    half_size: f64,
}

impl Octree {
    const ROOT_INDEX: usize = 0;

    /// Construct a new [`Octree`].
    ///
    /// * `theta`: the maximum ratio of the size of a node to the distance to it's center of mass
    ///     allowed when deciding whether or not to approximate the acceleration provided from
    ///     bodies within that node
    /// * `epsilon`: softening parameter, prevents
    /// * `half_size`: the half-extent of the simulation space
    pub const fn new(theta: f64, epsilon: f64, half_size: f64) -> Self {
        Self {
            nodes: Vec::new(),
            parents: Vec::new(),
            theta_squared: theta * theta,
            epsilon_squared: epsilon * epsilon,
            half_size,
        }
    }

    /// Clears all nodes from the [`Octree`].
    ///
    /// Automatically called with [`Octree::build`].
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.parents.clear();
    }

    /// True if there are no nodes in the [`Octree`].
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The number of nodes in the [`Octree`].
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[rustfmt::skip]
    #[cfg(not(feature = "multi-thread"))]
    fn subdivide(&mut self, node: usize, bodies: &mut [Body], range: Range<usize>) {
        use std::iter;

        let center = self.nodes[node].octant.center;

        const START: usize = 0;

        const SPLIT_X_1: usize = 1;
        const SPLIT_X_2: usize = 3;
        const SPLIT_X_3: usize = 5;
        const SPLIT_X_4: usize = 7;

        const SPLIT_Y_1: usize = 2;
        const SPLIT_Y_2: usize = 6;

        const SPLIT_Z_1: usize = 4;

        const END: usize = 8;

        let mut splits = [
            range.start,
            0, // 1 - 1st X split
            0, // 2 - 1st Y split
            0, // 3 - 2nd X split
            0, // 4 - 1st Z split
            0, // 5 - 3rd X split
            0, // 6 - 2nd Y split
            0, // 7 - 4th X split
            range.end,
        ];


        let predicate = |body: &Body| body.position.z < center.z;
        splits[SPLIT_Z_1] = splits[START] + partition(
            &mut bodies[splits[START]..splits[END]],
            predicate
        );

        let predicate = |body: &Body| body.position.y < center.y;
        splits[SPLIT_Y_1] = splits[START] + partition(
            &mut bodies[splits[START]..splits[SPLIT_Z_1]],
            predicate
        );
        splits[SPLIT_Y_2] = splits[SPLIT_Z_1] + partition(
            &mut bodies[splits[SPLIT_Z_1]..splits[END]],
            predicate
        );

        let predicate = |body: &Body| body.position.x < center.x;
        splits[SPLIT_X_1] = splits[START] + partition(
            &mut bodies[splits[START]..splits[SPLIT_Y_1]],
            predicate,
        );
        splits[SPLIT_X_2] = splits[SPLIT_Y_1] + partition(
            &mut bodies[splits[SPLIT_Y_1]..splits[SPLIT_Z_1]],
            predicate,
        );
        splits[SPLIT_X_3] = splits[SPLIT_Z_1] + partition(
            &mut bodies[splits[SPLIT_Z_1]..splits[SPLIT_Y_2]],
            predicate,
        );
        splits[SPLIT_X_4] = splits[SPLIT_Y_2] + partition(
            &mut bodies[splits[SPLIT_Y_2]..splits[END]],
            predicate,
        );

        self.parents.push(node);
        let children = NonZeroUsize::new(self.nodes.len()).expect("nodes must have at least 1 element");
        self.nodes[node].children = Some(children);

        let nexts = (1..8).map(|i| children.checked_add(i)).chain(iter::once(self.nodes[node].next));
        let octants = self.nodes[node].octant.subdivide();

        self.nodes.extend(
            splits
                .windows(2)
                .zip(nexts)
                .zip(octants)
                .map(|((bodies, next), octant)| Node::new(next, octant, bodies[0]..bodies[1]))
        );
    }

    #[cfg(feature = "multi-thread")]
    fn handle_node<'scope>(
        s: &Scope<'scope>,
        tx: crossbeam_channel::Sender<(usize, Node)>,
        next_index: &'scope AtomicUsize,
        octant: Octant,
        index: usize,
        next: Option<NonRootNodeIndex>,
        bodies: &'scope mut [Body],
        range: Range<usize>,
    ) {
        use std::sync::atomic::Ordering;

        let mut node = Node::new(next, octant, range.clone());

        if bodies.len() > LEAF_CAPACITY {
            let center = node.octant.center;

            const START: usize = 0;

            const SPLIT_X_1: usize = 1;
            const SPLIT_X_2: usize = 3;
            const SPLIT_X_3: usize = 5;
            const SPLIT_X_4: usize = 7;

            const SPLIT_Y_1: usize = 2;
            const SPLIT_Y_2: usize = 6;

            const SPLIT_Z_1: usize = 4;

            const END: usize = 8;

            let mut splits = [
                0,
                0, // 1 - 1st X split
                0, // 2 - 1st Y split
                0, // 3 - 2nd X split
                0, // 4 - 1st Z split
                0, // 5 - 3rd X split
                0, // 6 - 2nd Y split
                0, // 7 - 4th X split
                bodies.len(),
            ];

            let predicate = |body: &Body| body.position.z < center.z;
            splits[SPLIT_Z_1] =
                splits[START] + partition(&mut bodies[splits[START]..splits[END]], predicate);

            let predicate = |body: &Body| body.position.y < center.y;
            splits[SPLIT_Y_1] =
                splits[START] + partition(&mut bodies[splits[START]..splits[SPLIT_Z_1]], predicate);
            splits[SPLIT_Y_2] = splits[SPLIT_Z_1]
                + partition(&mut bodies[splits[SPLIT_Z_1]..splits[END]], predicate);

            let predicate = |body: &Body| body.position.x < center.x;
            splits[SPLIT_X_1] =
                splits[START] + partition(&mut bodies[splits[START]..splits[SPLIT_Y_1]], predicate);
            splits[SPLIT_X_2] = splits[SPLIT_Y_1]
                + partition(&mut bodies[splits[SPLIT_Y_1]..splits[SPLIT_Z_1]], predicate);
            splits[SPLIT_X_3] = splits[SPLIT_Z_1]
                + partition(&mut bodies[splits[SPLIT_Z_1]..splits[SPLIT_Y_2]], predicate);
            splits[SPLIT_X_4] = splits[SPLIT_Y_2]
                + partition(&mut bodies[splits[SPLIT_Y_2]..splits[END]], predicate);

            let children_index = next_index.fetch_add(8, Ordering::Relaxed);

            let nexts = [
                NonZeroUsize::new(children_index + 1),
                NonZeroUsize::new(children_index + 2),
                NonZeroUsize::new(children_index + 3),
                NonZeroUsize::new(children_index + 4),
                NonZeroUsize::new(children_index + 5),
                NonZeroUsize::new(children_index + 6),
                NonZeroUsize::new(children_index + 7),
                node.next,
            ];
            let ranges = [
                splits[0]..splits[1],
                splits[1]..splits[2],
                splits[2]..splits[3],
                splits[3]..splits[4],
                splits[4]..splits[5],
                splits[5]..splits[6],
                splits[6]..splits[7],
                splits[7]..splits[8],
            ];
            let octants = node.octant.subdivide();
            let bodies = bodies
                .get_disjoint_mut(ranges.clone())
                .expect("splits should be disjoint");

            node.children = Some(NonZeroUsize::new(children_index).expect("children is root"));

            tx.send((index, node)).expect("channel closed");

            for (i, ((octant, next), (local_range, bodies))) in (children_index..children_index + 8)
                .zip((octants.into_iter().zip(nexts)).zip(ranges.into_iter().zip(bodies)))
            {
                let tx = tx.clone();

                // The range given is shifted by the range passed in
                let range = (range.start + local_range.start)..(range.start + local_range.end);

                s.spawn(move |s| {
                    Self::handle_node(s, tx, next_index, octant, i, next, bodies, range)
                });
            }
        } else {
            // Compute center of mass for this node.
            for body in bodies {
                // Center of mass remains weighted by the total mass, will be divided later.
                node.center_of_mass += body.position * body.mass;
                node.mass += body.mass;
            }

            tx.send((index, node)).expect("channel closed");
        }
    }

    /// Clears and rebuilds this [`Octree`] from the bodies passed in.
    ///
    /// This method rearranges `bodies` such that each of the octree's nodes is able to index a
    /// slice of bodies belonging to it.
    ///
    /// * `bodies`: the bodies to rebuild with
    #[cfg(feature = "multi-thread")]
    pub fn build(&mut self, bodies: &mut [Body]) {
        use std::iter::repeat_n;
        use std::sync::atomic::AtomicUsize;

        use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

        self.clear();

        // We can expect to need at least bodies.len() / LEAF_CAPACITY new nodes for the leaf
        // nodes.
        self.nodes.reserve(bodies.len() / LEAF_CAPACITY);

        let root_octant = Octant {
            center: DVec3::ZERO,
            half_size: self.half_size,
        };

        let next_index = AtomicUsize::new(1);

        let (tx, rx) = crossbeam_channel::bounded::<(usize, Node)>(8);

        rayon::scope(|s| {
            s.spawn(|_| {
                while let Ok((index, node)) = rx.recv() {
                    self.nodes.extend(repeat_n(
                        Node::empty(),
                        (index + 1).saturating_sub(self.nodes.len()),
                    ));

                    if node.children.is_some() {
                        self.parents.push(index);
                    }

                    self.nodes[index] = node;
                }
            });

            s.spawn(|s| {
                Self::handle_node(
                    s,
                    tx,
                    &next_index,
                    root_octant,
                    Self::ROOT_INDEX,
                    None,
                    bodies,
                    0..bodies.len(),
                );
            });
        });

        // Here we propagate the centers of mass up the tree.
        for &index in self.parents.iter().rev() {
            let Some(children_range) = self.nodes[index].children_range() else {
                unreachable!();
            };

            let mut center_of_mass = DVec3::ZERO;
            let mut mass = 0.0;

            for child in &self.nodes[children_range] {
                center_of_mass += child.center_of_mass;
                mass += child.mass;
            }

            self.nodes[index].center_of_mass = center_of_mass;
            self.nodes[index].mass = mass;
        }

        self.nodes.par_iter_mut().for_each(|node| {
            if node.mass != 0.0 {
                node.center_of_mass /= node.mass;
            }
        });
    }

    /// Clears and rebuilds this [`Octree`] from the bodies passed in.
    ///
    /// This method rearranges `bodies` such that each of the octree's nodes is able to index a
    /// slice of bodies belonging to it.
    ///
    /// * `bodies`: the bodies to rebuild with
    #[cfg(not(feature = "multi-thread"))]
    pub fn build(&mut self, bodies: &mut [Body]) {
        self.clear();

        // We can expect to need at least bodies.len() / LEAF_CAPACITY new nodes for the leaf
        // nodes.
        self.nodes.reserve(bodies.len() / LEAF_CAPACITY);

        self.nodes.push(Node::new(
            None,
            Octant {
                center: DVec3::ZERO,
                half_size: self.half_size,
            },
            0..bodies.len(),
        ));

        let mut node = 0;
        while node < self.nodes.len() {
            let range = self.nodes[node].bodies.clone();

            if range.len() > LEAF_CAPACITY {
                self.subdivide(node, bodies, range);
            } else {
                // Compute center of mass for this node.
                let node = &mut self.nodes[node];

                for body in &bodies[range] {
                    // Center of mass remains weighted by the total mass, will be divided later.
                    node.center_of_mass += body.position * body.mass;
                    node.mass += body.mass;
                }
            }

            node += 1;
        }

        // Here we propagate the centers of mass up the tree.
        for &index in self.parents.iter().rev() {
            let Some(children_range) = self.nodes[index].children_range() else {
                unreachable!();
            };

            let mut center_of_mass = DVec3::ZERO;
            let mut mass = 0.0;

            for child in &self.nodes[children_range] {
                center_of_mass += child.center_of_mass;
                mass += child.mass;
            }

            self.nodes[index].center_of_mass = center_of_mass;
            self.nodes[index].mass = mass;
        }

        for node in self.nodes.iter_mut() {
            if node.mass != 0.0 {
                node.center_of_mass /= node.mass;
            }
        }
    }

    /// Computes the acceleration due to gravity at a given position.
    ///
    /// * `position`: the position of the body experiencing the acceleration
    /// * `bodies`: the set of all bodies
    /// * `ignore_body`: the id of the body to compute the acceleration for, ignores it in computations
    #[must_use]
    pub fn acceleration(
        &self,
        position: DVec3,
        bodies: &[Body],
        ignore_body: Option<BodyId>,
    ) -> DVec3 {
        let mut acceleration = DVec3::ZERO;

        let mut index = Self::ROOT_INDEX;

        loop {
            let node = &self.nodes[index];

            let direction =
                periodic_boundary_distance_direction(position, node.center_of_mass, self.half_size);
            let distance_squared = direction.length_squared();

            // Performs the check "size / theta < theta" to determine if this node is good enough
            // to stop at.
            if (2.0 * node.octant.half_size).powi(2) < distance_squared * self.theta_squared {
                if direction != DVec3::ZERO {
                    acceleration += direction.normalize()
                        * (node.mass / (distance_squared + self.epsilon_squared));
                }

                match node.next {
                    Some(next) => index = next.into(),
                    None => return acceleration,
                }
            } else if let Some(children) = node.children {
                index = children.into();
            } else {
                for body in &bodies[node.bodies.clone()] {
                    if ignore_body.is_some_and(|id| body.id == id) {
                        continue;
                    }

                    let direction = periodic_boundary_distance_direction(
                        position,
                        body.position,
                        self.half_size,
                    );

                    if direction == DVec3::ZERO {
                        continue;
                    }

                    let distance_squared = direction.length_squared();

                    acceleration += direction.normalize()
                        * (body.mass / (distance_squared + self.epsilon_squared));
                }

                match node.next {
                    Some(next) => index = next.into(),
                    None => return acceleration,
                }
            }
        }
    }
}
