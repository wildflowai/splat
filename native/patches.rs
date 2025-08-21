use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
#[pyclass]
pub struct BoundingBox {
    #[pyo3(get, set)]
    pub min_x: f64,
    #[pyo3(get, set)]
    pub max_x: f64,
    #[pyo3(get, set)]
    pub min_y: f64,
    #[pyo3(get, set)]
    pub max_y: f64,
}

#[pymethods]
impl BoundingBox {
    #[new]
    fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        BoundingBox {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    #[getter]
    fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    #[getter]
    fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
}

#[derive(Debug, Clone, Copy)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }
}

#[derive(Debug, Clone)]
struct DpState {
    patch_count: usize,
    aspect_cost: f64,
    patches: Vec<BoundingBox>,
}

impl DpState {
    fn infinity() -> Self {
        DpState {
            patch_count: 999999,
            aspect_cost: f64::INFINITY,
            patches: Vec::new(),
        }
    }

    fn empty() -> Self {
        DpState {
            patch_count: 0,
            aspect_cost: 1.0,
            patches: Vec::new(),
        }
    }

    fn add(&self, other: &DpState) -> Self {
        let mut patches = self.patches.clone();
        patches.extend(other.patches.clone());
        DpState {
            patch_count: self.patch_count + other.patch_count,
            aspect_cost: self.aspect_cost * other.aspect_cost,
            patches,
        }
    }

    fn is_better_than(&self, other: &DpState) -> bool {
        if self.patch_count != other.patch_count {
            return self.patch_count < other.patch_count;
        }
        self.aspect_cost < other.aspect_cost
    }
}

struct Binner {
    width: f64,
    start: f64,
    count: usize,
}

impl Binner {
    fn new(min_val: f64, max_val: f64, target_bins: usize) -> Self {
        if max_val <= min_val {
            panic!("Invalid range: [{}, {}]", min_val, max_val);
        }

        let rough = (max_val - min_val) / target_bins as f64;
        let power = 10.0_f64.powf(rough.log10().floor());
        let nice_numbers = [1.0, 2.0, 5.0, 10.0];
        let nice = nice_numbers
            .iter()
            .find(|&&x| x * power >= rough)
            .unwrap_or(&10.0)
            * power;

        let start = (min_val / nice).floor() * nice;
        let count = ((max_val - start) / nice).ceil() as usize;

        Binner {
            width: nice,
            start,
            count,
        }
    }

    fn bin_id(&self, x: f64) -> usize {
        let id = ((x - self.start) / self.width) as usize;
        id.min(self.count.saturating_sub(1))
    }

    fn bounds(&self, i: usize) -> (f64, f64) {
        (
            self.start + i as f64 * self.width,
            self.start + (i + 1) as f64 * self.width,
        )
    }
}

struct FastCounter {
    cameras: Vec<Point2D>,
    left: usize,
    right: usize,
    last_y_start: f64,
}

impl FastCounter {
    fn new(mut cameras_sorted_by_y: Vec<Point2D>) -> Self {
        cameras_sorted_by_y.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal));
        FastCounter {
            cameras: cameras_sorted_by_y,
            left: 0,
            right: 0,
            last_y_start: f64::NEG_INFINITY,
        }
    }

    fn count_in_y_range(&mut self, y_start: f64, y_end: f64) -> usize {
        // Reset if going backwards
        if y_start < self.last_y_start {
            self.left = 0;
            self.right = 0;
        }

        // Move left pointer to first camera >= y_start
        while self.left < self.cameras.len() && self.cameras[self.left].y < y_start {
            self.left += 1;
        }

        // Move right pointer to first camera > y_end
        while self.right < self.cameras.len() && self.cameras[self.right].y <= y_end {
            self.right += 1;
        }

        self.last_y_start = y_start;
        self.right.saturating_sub(self.left)
    }
}

fn pack_y_direction(
    cameras_in_x_range: &[Point2D],
    all_cameras: &[Point2D],
    x_left: f64,
    x_right: f64,
    y_binner: &Binner,
    buffer_m: f64,
    max_cameras: usize,
) -> DpState {
    if cameras_in_x_range.is_empty() {
        return DpState::empty();
    }

    // Find Y range needed
    let min_y = cameras_in_x_range
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = cameras_in_x_range
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_bin = y_binner.bin_id(min_y);
    let max_bin = y_binner.bin_id(max_y);

    // Pre-filter cameras in buffered X range
    let buffered_cameras: Vec<Point2D> = all_cameras
        .iter()
        .filter(|cam| cam.x >= x_left - buffer_m && cam.x <= x_right + buffer_m)
        .cloned()
        .collect();

    let mut fast_counter = if !buffered_cameras.is_empty() {
        Some(FastCounter::new(buffered_cameras))
    } else {
        None
    };

    let mut patches = Vec::new();
    let mut current_bin = min_bin;

    // Greedily create patches
    while current_bin <= max_bin {
        let mut best_end_bin = None;

        // Find largest Y range that fits camera constraint
        for end_bin in current_bin..=max_bin {
            let y_start = y_binner.bounds(current_bin).0;
            let y_end = y_binner.bounds(end_bin).1;

            // Count cameras in buffered Y range
            let camera_count = if let Some(ref mut counter) = fast_counter {
                counter.count_in_y_range(y_start - buffer_m, y_end + buffer_m)
            } else {
                0
            };

            if camera_count <= max_cameras {
                best_end_bin = Some(end_bin);
            } else {
                break;
            }
        }

        if best_end_bin.is_none() {
            // Handle clustering edge cases
            let y_groups = group_cameras_by_y(cameras_in_x_range);

            if should_use_special_clustering(&y_groups, max_cameras) {
                return handle_clustering_edge_case(
                    &y_groups,
                    x_left,
                    x_right,
                    buffer_m,
                    max_cameras,
                );
            }

            return DpState::infinity();
        }

        let end_bin = best_end_bin.unwrap();
        let y_start = y_binner.bounds(current_bin).0;
        let y_end = y_binner.bounds(end_bin).1;

        let patch = BoundingBox::new(
            x_left - buffer_m,
            x_right + buffer_m,
            y_start - buffer_m,
            y_end + buffer_m,
        );

        patches.push(patch);
        current_bin = end_bin + 1;
    }

    // Calculate aspect ratio cost
    let mut total_cost = 1.0;
    for patch in &patches {
        let aspect_ratio = patch.width().max(patch.height()) / patch.width().min(patch.height());
        total_cost *= aspect_ratio;
    }

    DpState {
        patch_count: patches.len(),
        aspect_cost: total_cost,
        patches,
    }
}

fn group_cameras_by_y(cameras: &[Point2D]) -> HashMap<i64, Vec<Point2D>> {
    let mut groups = HashMap::new();
    for &camera in cameras {
        let y_key = (camera.y * 1000.0).round() as i64;
        groups.entry(y_key).or_insert_with(Vec::new).push(camera);
    }
    groups
}

fn should_use_special_clustering(
    y_groups: &HashMap<i64, Vec<Point2D>>,
    max_cameras: usize,
) -> bool {
    let has_oversized_group = y_groups.values().any(|group| group.len() > max_cameras);
    if !has_oversized_group {
        return false;
    }

    // Check for dense clustering
    for group in y_groups.values() {
        if group.len() > max_cameras {
            let x_coords: Vec<f64> = group.iter().map(|c| c.x).collect();
            let x_span = x_coords.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - x_coords.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let density = group.len() as f64 / x_span.max(0.1);

            if density > 10.0 {
                return true;
            }
        }
    }
    false
}

fn handle_clustering_edge_case(
    y_groups: &HashMap<i64, Vec<Point2D>>,
    _x_left: f64,
    _x_right: f64,
    buffer_m: f64,
    max_cameras: usize,
) -> DpState {
    let mut special_patches = Vec::new();

    for group in y_groups.values() {
        if group.len() > max_cameras {
            let y_val = group[0].y;
            let min_x = group.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
            let max_x = group.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);

            let x_patches =
                partition_1d_horizontal(group, max_cameras, buffer_m, y_val, min_x, max_x);
            special_patches.extend(x_patches);
        } else {
            let min_x = group.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
            let max_x = group.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);
            let y_val = group[0].y;

            let patch = BoundingBox::new(
                min_x - buffer_m,
                max_x + buffer_m,
                y_val - buffer_m,
                y_val + buffer_m,
            );
            special_patches.push(patch);
        }
    }

    // Calculate aspect ratio cost
    let mut total_cost = 1.0;
    for patch in &special_patches {
        let aspect_ratio = patch.width().max(patch.height()) / patch.width().min(patch.height());
        total_cost *= aspect_ratio;
    }

    DpState {
        patch_count: special_patches.len(),
        aspect_cost: total_cost,
        patches: special_patches,
    }
}

fn partition_1d_horizontal(
    cameras: &[Point2D],
    max_cameras: usize,
    buffer_meters: f64,
    fixed_y: f64,
    _min_x: f64,
    _max_x: f64,
) -> Vec<BoundingBox> {
    let mut cameras_by_x = cameras.to_vec();
    cameras_by_x.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

    let mut patches = Vec::new();
    let min_height = (buffer_meters * 2.0).max(0.1);

    let mut i = 0;
    while i < cameras_by_x.len() {
        let end_idx = (i + max_cameras).min(cameras_by_x.len());
        let patch_cameras = &cameras_by_x[i..end_idx];

        let patch_min_x = patch_cameras[0].x;
        let patch_max_x = patch_cameras[patch_cameras.len() - 1].x;

        let center_x = (patch_min_x + patch_max_x) / 2.0;
        let width_with_buffer = (patch_max_x - patch_min_x + 2.0 * buffer_meters).max(0.1);

        patches.push(BoundingBox::new(
            center_x - width_with_buffer / 2.0,
            center_x + width_with_buffer / 2.0,
            fixed_y - min_height / 2.0,
            fixed_y + min_height / 2.0,
        ));

        i += max_cameras;
    }

    patches
}

fn partition_1d_vertical(
    cameras: &[Point2D],
    max_cameras: usize,
    buffer_meters: f64,
    fixed_x: f64,
    _min_y: f64,
    _max_y: f64,
) -> Vec<BoundingBox> {
    let mut cameras_by_y = cameras.to_vec();
    cameras_by_y.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal));

    let mut patches = Vec::new();
    let min_width = (buffer_meters * 2.0).max(0.1);

    let mut i = 0;
    while i < cameras_by_y.len() {
        let end_idx = (i + max_cameras).min(cameras_by_y.len());
        let patch_cameras = &cameras_by_y[i..end_idx];

        let patch_min_y = patch_cameras[0].y;
        let patch_max_y = patch_cameras[patch_cameras.len() - 1].y;

        let center_y = (patch_min_y + patch_max_y) / 2.0;
        let height_with_buffer = (patch_max_y - patch_min_y + 2.0 * buffer_meters).max(0.1);

        patches.push(BoundingBox::new(
            fixed_x - min_width / 2.0,
            fixed_x + min_width / 2.0,
            center_y - height_with_buffer / 2.0,
            center_y + height_with_buffer / 2.0,
        ));

        i += max_cameras;
    }

    patches
}

fn partition_with_dp(
    camera_positions: Vec<(f64, f64)>,
    max_cameras: usize,
    buffer_meters: f64,
    target_bins: usize,
) -> PyResult<Vec<BoundingBox>> {
    if camera_positions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Camera positions cannot be empty",
        ));
    }

    // Convert to internal format
    let cameras: Vec<Point2D> = camera_positions
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let mut cameras_by_x = cameras.clone();
    cameras_by_x.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

    // Calculate bounds
    let min_x = cameras.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
    let max_x = cameras
        .iter()
        .map(|c| c.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = cameras.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
    let max_y = cameras
        .iter()
        .map(|c| c.y)
        .fold(f64::NEG_INFINITY, f64::max);

    // Handle edge case: all cameras at same position
    if (min_x - max_x).abs() < f64::EPSILON && (min_y - max_y).abs() < f64::EPSILON {
        return Ok(vec![BoundingBox::new(
            min_x - buffer_meters,
            max_x + buffer_meters,
            min_y - buffer_meters,
            max_y + buffer_meters,
        )]);
    }

    // Handle 1D cases
    if (min_x - max_x).abs() < f64::EPSILON && cameras.len() <= max_cameras {
        return Ok(vec![BoundingBox::new(
            min_x - buffer_meters,
            max_x + buffer_meters,
            min_y - buffer_meters,
            max_y + buffer_meters,
        )]);
    }

    if (min_y - max_y).abs() < f64::EPSILON && cameras.len() <= max_cameras {
        return Ok(vec![BoundingBox::new(
            min_x - buffer_meters,
            max_x + buffer_meters,
            min_y - buffer_meters,
            max_y + buffer_meters,
        )]);
    }

    // Handle 1D cases that exceed max_cameras
    if (min_x - max_x).abs() < f64::EPSILON {
        return Ok(partition_1d_vertical(
            &cameras,
            max_cameras,
            buffer_meters,
            min_x,
            min_y,
            max_y,
        ));
    }

    if (min_y - max_y).abs() < f64::EPSILON {
        return Ok(partition_1d_horizontal(
            &cameras,
            max_cameras,
            buffer_meters,
            min_y,
            min_x,
            max_x,
        ));
    }

    // Normal 2D case - create binners
    let x_binner = Binner::new(min_x, max_x, target_bins);
    let y_binner = Binner::new(min_y, max_y, target_bins);

    // Dynamic programming on X axis
    let mut dp = vec![DpState::infinity(); x_binner.count];

    for cur_bin in 0..x_binner.count {
        for prev_bin in 0..=cur_bin {
            // X range for this partition
            let x_left = x_binner.bounds(prev_bin).0;
            let x_right = x_binner.bounds(cur_bin).1;

            // Get cameras in this X range
            let cameras_in_range: Vec<Point2D> = if cur_bin == x_binner.count - 1 {
                cameras_by_x
                    .iter()
                    .filter(|c| c.x >= x_left && c.x <= x_right)
                    .cloned()
                    .collect()
            } else {
                cameras_by_x
                    .iter()
                    .filter(|c| c.x >= x_left && c.x < x_right)
                    .cloned()
                    .collect()
            };

            // Pack Y direction for this X range
            let new_state = pack_y_direction(
                &cameras_in_range,
                &cameras,
                x_left,
                x_right,
                &y_binner,
                buffer_meters,
                max_cameras,
            );

            // Combine with previous solution
            let combined_state = if prev_bin > 0 {
                dp[prev_bin - 1].add(&new_state)
            } else {
                new_state
            };

            // Keep if better
            if combined_state.is_better_than(&dp[cur_bin]) {
                dp[cur_bin] = combined_state;
            }
        }
    }

    let final_solution = &dp[x_binner.count - 1];

    if final_solution.patch_count >= 999999 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No solution found - try increasing max_cameras or reducing buffer",
        ));
    }

    println!(
        "✅ Created {} patches (aspect cost: {:.2})",
        final_solution.patch_count, final_solution.aspect_cost
    );

    Ok(final_solution.patches.clone())
}

#[pyfunction]
pub fn patches(
    camera_positions: Vec<(f64, f64)>,
    max_cameras: Option<usize>,
    buffer_meters: Option<f64>,
    target_bins: Option<usize>,
) -> PyResult<Vec<BoundingBox>> {
    partition_with_dp(
        camera_positions,
        max_cameras.unwrap_or(700),
        buffer_meters.unwrap_or(1.5),
        target_bins.unwrap_or(100),
    )
}
