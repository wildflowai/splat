use byteorder::{LittleEndian, ReadBytesExt};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::time::Instant;

fn default_min_neighbors() -> usize {
    10
}
fn default_radius() -> f64 {
    0.1
}

/// Configuration for cleaning a PLY point cloud.
///
/// This struct holds all the parameters that control the behavior of the cleanup process,
/// including input/output file paths, and various filtering criteria.
#[derive(Debug, Deserialize, Serialize, Clone)]
struct CleanConfig {
    pub input_file: String,
    pub output_file: String,
    #[serde(default)]
    pub discarded_file: Option<String>,
    #[serde(default)]
    pub min_x: Option<f64>,
    #[serde(default)]
    pub min_y: Option<f64>,
    #[serde(default)]
    pub min_z: Option<f64>,
    #[serde(default)]
    pub max_x: Option<f64>,
    #[serde(default)]
    pub max_y: Option<f64>,
    #[serde(default)]
    pub max_z: Option<f64>,
    pub max_area: f64,
    #[serde(default)]
    pub colmap_points_file: Option<String>,
    #[serde(default = "default_min_neighbors")]
    pub min_neighbors: usize,
    #[serde(default = "default_radius")]
    pub radius: f64,
}

#[derive(Debug, Clone, Copy)]
struct PropertyLayout {
    x_offset: usize,
    y_offset: usize,
    z_offset: usize,
    scale_0_offset: usize,
    scale_1_offset: usize,
    scale_2_offset: usize,
    row_size: usize,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct GridCell {
    x: i32,
    y: i32,
    z: i32,
}

fn point_to_grid_cell(point: &[f64; 3], cell_size: f64) -> GridCell {
    GridCell {
        x: (point[0] / cell_size).floor() as i32,
        y: (point[1] / cell_size).floor() as i32,
        z: (point[2] / cell_size).floor() as i32,
    }
}

fn read_colmap_points3d_binary_to_grid(
    path: &str,
    cell_size: f64,
) -> Result<HashMap<GridCell, Vec<[f64; 3]>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let num_points = reader.read_u64::<LittleEndian>()?;

    let mut grid: HashMap<GridCell, Vec<[f64; 3]>> = HashMap::new();

    for _ in 0..num_points {
        let _point_id = reader.read_u64::<LittleEndian>()?;
        let x = reader.read_f64::<LittleEndian>()?;
        let y = reader.read_f64::<LittleEndian>()?;
        let z = reader.read_f64::<LittleEndian>()?;
        let _r = reader.read_u8()?;
        let _g = reader.read_u8()?;
        let _b = reader.read_u8()?;
        let _error = reader.read_f64::<LittleEndian>()?;
        let track_len = reader.read_u64::<LittleEndian>()?;
        
        // Skip track data: each track element is 8 bytes (image_id: u32 + point2d_idx: u32)
        let mut skip_buf = vec![0u8; (track_len * 8) as usize];
        reader.read_exact(&mut skip_buf)?;

        let point = [x, y, z];
        let cell = point_to_grid_cell(&point, cell_size);
        grid.entry(cell).or_insert_with(Vec::new).push(point);
    }

    Ok(grid)
}

fn read_ply_header(
    reader: &mut BufReader<&mut File>,
) -> Result<(usize, usize, String, PropertyLayout), Box<dyn std::error::Error>> {
    let mut header = String::new();
    let mut header_size = 0;
    let mut vertex_count = 0;
    let mut properties = Vec::new();

    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        header.push_str(&line);
        header_size += bytes_read;
        let line_trimmed = line.trim();

        if line_trimmed == "end_header" {
            break;
        } else if line_trimmed.starts_with("element vertex") {
            vertex_count = line_trimmed
                .split_whitespace()
                .nth(2)
                .unwrap_or("0")
                .parse()?;
        } else if line_trimmed.starts_with("property") {
            let parts: Vec<&str> = line_trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                properties.push((parts[1].to_string(), parts[2].to_string()));
            }
        }
    }

    let mut layout = PropertyLayout {
        x_offset: 0,
        y_offset: 0,
        z_offset: 0,
        scale_0_offset: 0,
        scale_1_offset: 0,
        scale_2_offset: 0,
        row_size: 0,
    };

    let mut offset = 0;
    for (prop_type, prop_name) in &properties {
        let size = match prop_type.as_str() {
            "float" | "f32" => 4,
            "double" | "f64" => 8,
            "uchar" | "u8" => 1,
            "int" | "i32" => 4,
            _ => 4,
        };

        match prop_name.as_str() {
            "x" => layout.x_offset = offset,
            "y" => layout.y_offset = offset,
            "z" => layout.z_offset = offset,
            "scale_0" => layout.scale_0_offset = offset,
            "scale_1" => layout.scale_1_offset = offset,
            "scale_2" => layout.scale_2_offset = offset,
            _ => {}
        }
        offset += size;
    }
    layout.row_size = offset;

    Ok((vertex_count, header_size, header, layout))
}

fn write_ply_data(
    writer: &mut BufWriter<File>,
    header: &str,
    original_vertex_count: usize,
    new_vertex_count: usize,
    vertex_data: &[u8],
) -> Result<(), std::io::Error> {
    let new_header = header.replace(
        &format!("element vertex {}", original_vertex_count),
        &format!("element vertex {}", new_vertex_count),
    );
    writer.write_all(new_header.as_bytes())?;
    writer.write_all(vertex_data)?;
    Ok(())
}

fn cleanup_ply(config: &CleanConfig) -> PyResult<()> {
    let start_time = Instant::now();
    println!("Starting PLY cleanup process...");

    if config.radius == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Radius cannot be 0.0. Please provide a positive radius value.",
        ));
    }

    let colmap_grid = if let Some(colmap_points_file) = &config.colmap_points_file {
        println!(
            "Reading COLMAP points from {} and building grid...",
            colmap_points_file
        );
        let colmap_read_start = Instant::now();
        let grid = read_colmap_points3d_binary_to_grid(colmap_points_file, config.radius)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        println!(
            "Finished reading COLMAP points and building grid in {:?}",
            colmap_read_start.elapsed()
        );
        Some(Arc::new(grid))
    } else {
        None
    };

    let mut file = File::open(&config.input_file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let mut reader = BufReader::new(&mut file);

    println!("Reading PLY header from {}...", config.input_file);
    let header_read_start = Instant::now();
    let (vertex_count, header_size, header, layout) = read_ply_header(&mut reader)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    println!(
        "Finished reading PLY header in {:?}. Found {} vertices.",
        header_read_start.elapsed(),
        vertex_count
    );

    println!("Reading PLY vertex data...");
    let vertex_data_read_start = Instant::now();
    file.seek(SeekFrom::Start(header_size as u64)).unwrap();
    let mut vertex_data = vec![0u8; layout.row_size * vertex_count];
    file.read_exact(&mut vertex_data).unwrap();
    println!(
        "Finished reading PLY vertex data in {:?}",
        vertex_data_read_start.elapsed()
    );

    println!("Starting splat filtering (parallelized)...");
    let filtering_start = Instant::now();

    let config_arc = Arc::new(config.clone());
    let vertex_data_arc = Arc::new(vertex_data);

    let results: Vec<bool> =
        (0..vertex_count)
            .into_par_iter()
            .map(|i| {
                let row_start = i * layout.row_size;
                let row_data = &vertex_data_arc[row_start..row_start + layout.row_size];

                let x = f32::from_le_bytes([
                    row_data[layout.x_offset],
                    row_data[layout.x_offset + 1],
                    row_data[layout.x_offset + 2],
                    row_data[layout.x_offset + 3],
                ]) as f64;
                let y = f32::from_le_bytes([
                    row_data[layout.y_offset],
                    row_data[layout.y_offset + 1],
                    row_data[layout.y_offset + 2],
                    row_data[layout.y_offset + 3],
                ]) as f64;
                let z = f32::from_le_bytes([
                    row_data[layout.z_offset],
                    row_data[layout.z_offset + 1],
                    row_data[layout.z_offset + 2],
                    row_data[layout.z_offset + 3],
                ]) as f64;

                let s0 = f32::from_le_bytes([
                    row_data[layout.scale_0_offset],
                    row_data[layout.scale_0_offset + 1],
                    row_data[layout.scale_0_offset + 2],
                    row_data[layout.scale_0_offset + 3],
                ]) as f64;
                let s1 = f32::from_le_bytes([
                    row_data[layout.scale_1_offset],
                    row_data[layout.scale_1_offset + 1],
                    row_data[layout.scale_1_offset + 2],
                    row_data[layout.scale_1_offset + 3],
                ]) as f64;
                let s2 = f32::from_le_bytes([
                    row_data[layout.scale_2_offset],
                    row_data[layout.scale_2_offset + 1],
                    row_data[layout.scale_2_offset + 2],
                    row_data[layout.scale_2_offset + 3],
                ]) as f64;

                let area = s0.exp().powi(2) + s1.exp().powi(2) + s2.exp().powi(2);

                let mut keep = true;

                if let Some(grid) = &colmap_grid {
                    let splat_point = [x, y, z];
                    let splat_cell = point_to_grid_cell(&splat_point, config_arc.radius);
                    let mut neighbors_count = 0;
                    let radius_sq = config_arc.radius.powi(2);

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let neighbor_cell = GridCell {
                                    x: splat_cell.x + dx,
                                    y: splat_cell.y + dy,
                                    z: splat_cell.z + dz,
                                };
                                if let Some(points_in_cell) = grid.get(&neighbor_cell) {
                                    for colmap_point in points_in_cell {
                                        let dist_sq = (colmap_point[0] - splat_point[0]).powi(2)
                                            + (colmap_point[1] - splat_point[1]).powi(2)
                                            + (colmap_point[2] - splat_point[2]).powi(2);
                                        if dist_sq <= radius_sq {
                                            neighbors_count += 1;
                                            if neighbors_count >= config_arc.min_neighbors {
                                                break;
                                            }
                                        }
                                    }
                                }
                                if neighbors_count >= config_arc.min_neighbors {
                                    break;
                                }
                            }
                            if neighbors_count >= config_arc.min_neighbors {
                                break;
                            }
                        }
                        if neighbors_count >= config_arc.min_neighbors {
                            break;
                        }
                    }

                    if neighbors_count < config_arc.min_neighbors {
                        keep = false;
                    }
                }

                if let (
                    Some(min_x),
                    Some(min_y),
                    Some(min_z),
                    Some(max_x),
                    Some(max_y),
                    Some(max_z),
                ) = (
                    config_arc.min_x,
                    config_arc.min_y,
                    config_arc.min_z,
                    config_arc.max_x,
                    config_arc.max_y,
                    config_arc.max_z,
                ) {
                    if !(x >= min_x
                        && x < max_x
                        && y >= min_y
                        && y < max_y
                        && z >= min_z
                        && z < max_z)
                    {
                        keep = false;
                    }
                }

                if area >= config_arc.max_area {
                    keep = false;
                }
                keep
            })
            .collect();

    let mut final_kept_vertices = Vec::new();
    let mut final_discarded_vertices = Vec::new();

    for (i, &keep) in results.iter().enumerate() {
        let row_start = i * layout.row_size;
        let row_end = row_start + layout.row_size;
        if keep {
            final_kept_vertices.extend_from_slice(&vertex_data_arc[row_start..row_end]);
        } else {
            final_discarded_vertices.extend_from_slice(&vertex_data_arc[row_start..row_end]);
        }
    }

    println!(
        "Finished splat filtering in {:?}",
        filtering_start.elapsed()
    );

    println!("Writing kept splats to {}...", config.output_file);
    let write_kept_start = Instant::now();
    let output_file = File::create(&config.output_file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let mut writer = BufWriter::new(output_file);
    write_ply_data(
        &mut writer,
        &header,
        vertex_count,
        final_kept_vertices.len() / layout.row_size,
        &final_kept_vertices,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    println!(
        "Finished writing kept splats in {:?}. Kept {} splats.",
        write_kept_start.elapsed(),
        final_kept_vertices.len() / layout.row_size
    );

    if let Some(discarded_file_path) = &config.discarded_file {
        println!("Writing discarded splats to {}...", discarded_file_path);
        let write_discarded_start = Instant::now();
        let discarded_file = File::create(discarded_file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut discarded_writer = BufWriter::new(discarded_file);
        write_ply_data(
            &mut discarded_writer,
            &header,
            vertex_count,
            final_discarded_vertices.len() / layout.row_size,
            &final_discarded_vertices,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        println!(
            "Finished writing discarded splats in {:?}. Discarded {} splats.",
            write_discarded_start.elapsed(),
            final_discarded_vertices.len() / layout.row_size
        );
    }

    println!(
        "PLY cleanup process completed in {:?}",
        start_time.elapsed()
    );
    Ok(())
}

#[pyfunction]
pub fn cleanup_splats(config_json: &str) -> PyResult<()> {
    let config: CleanConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    cleanup_ply(&config)
}
