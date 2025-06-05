use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write, Seek, SeekFrom, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use rand::{Rng, thread_rng};
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use indicatif::{ProgressBar, ProgressStyle};

// Default value functions for serde
fn default_sample_percentage() -> f64 { 100.0 }
fn default_neg_inf() -> f64 { f64::NEG_INFINITY }
fn default_pos_inf() -> f64 { f64::INFINITY }

#[derive(Debug, Deserialize, Serialize, Clone)]
#[pyclass]
pub struct Config {
    #[pyo3(get, set)]
    pub input_file: String,
    #[pyo3(get, set)]
    #[serde(default = "default_sample_percentage")]
    pub sample_percentage: f64,
    #[pyo3(get, set)]
    #[serde(rename = "minZ", default = "default_neg_inf")]
    pub min_z: f64,
    #[pyo3(get, set)]
    #[serde(rename = "maxZ", default = "default_pos_inf")]
    pub max_z: f64,
    #[pyo3(get, set)]
    pub patches: Vec<Patch>,
}

#[pymethods]
impl Config {
    #[new]
    fn new(input_file: String) -> Self {
        Config {
            input_file,
            sample_percentage: 100.0,
            min_z: f64::NEG_INFINITY,
            max_z: f64::INFINITY,
            patches: Vec::new(),
        }
    }

    fn add_patch(&mut self, patch: Patch) {
        self.patches.push(patch);
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Config> {
        serde_json::from_str(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON parse error: {}", e)))
    }

    #[staticmethod]
    fn from_file(file_path: &str) -> PyResult<Config> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File read error: {}", e)))?;
        Self::from_json(&content)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialize error: {}", e)))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[pyclass]
pub struct Patch {
    #[pyo3(get, set)]
    #[serde(rename = "minX", default = "default_neg_inf")]
    pub min_x: f64,
    #[pyo3(get, set)]
    #[serde(rename = "minY", default = "default_neg_inf")]
    pub min_y: f64,
    #[pyo3(get, set)]
    #[serde(rename = "maxX", default = "default_pos_inf")]
    pub max_x: f64,
    #[pyo3(get, set)]
    #[serde(rename = "maxY", default = "default_pos_inf")]
    pub max_y: f64,
    #[pyo3(get, set)]
    pub output_file: String,
}

#[pymethods]
impl Patch {
    #[new]
    fn new(output_file: String) -> Self {
        Patch {
            min_x: f64::NEG_INFINITY,
            min_y: f64::NEG_INFINITY,
            max_x: f64::INFINITY,
            max_y: f64::INFINITY,
            output_file,
        }
    }

    #[classmethod]
    fn with_bounds(_cls: &Bound<'_, PyType>, output_file: String, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Patch {
            min_x,
            min_y,
            max_x,
            max_y,
            output_file,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Point3D {
    x: f64,
    y: f64,
    z: f64,
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug, Clone, Copy)]
struct PropertyLayout {
    x_offset: usize,
    y_offset: usize,
    z_offset: usize,
    r_offset: Option<usize>,
    g_offset: Option<usize>,
    b_offset: Option<usize>,
    row_size: usize,
}

pub fn read_ply_file(file_path: &str, config: &Config) -> Result<Vec<Point3D>, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut reader = BufReader::new(&mut file);
    
    let mut header_size = 0;
    let mut vertex_count = 0;
    let mut properties = Vec::new();
    
    // Parse header
    loop {
        let mut line = String::new();
        header_size += reader.read_line(&mut line)?;
        let line = line.trim();
        
        if line == "end_header" {
            break;
        } else if line.starts_with("element vertex ") {
            vertex_count = line.split_whitespace().nth(2).unwrap_or("0").parse()?;
        } else if line.starts_with("property ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                properties.push((parts[1].to_string(), parts[2].to_string()));
            }
        }
    }
    
    // Build property layout
    let mut layout = PropertyLayout {
        x_offset: 0, y_offset: 0, z_offset: 0,
        r_offset: None, g_offset: None, b_offset: None,
        row_size: 0,
    };
    
    let mut offset = 0;
    for (prop_type, prop_name) in &properties {
        let size = match prop_type.as_str() {
            "float" => 4, "double" => 8, "uchar" => 1, "int" => 4, _ => 4,
        };
        
        match prop_name.as_str() {
            "x" => layout.x_offset = offset,
            "y" => layout.y_offset = offset,
            "z" => layout.z_offset = offset,
            "red" | "r" => layout.r_offset = Some(offset),
            "green" | "g" => layout.g_offset = Some(offset),
            "blue" | "b" => layout.b_offset = Some(offset),
            _ => {}
        }
        offset += size;
    }
    layout.row_size = offset;
    
    // Read vertex data
    file.seek(SeekFrom::Start(header_size as u64))?;
    let mut vertex_data = vec![0u8; layout.row_size * vertex_count];
    file.read_exact(&mut vertex_data)?;
    
    // Extract points
    let mut points = Vec::new();
    let mut rng = thread_rng();
    let sample_threshold = config.sample_percentage / 100.0;
    
    // Progress bar for point extraction
    let pb = ProgressBar::new(vertex_count as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})").unwrap());
    pb.set_message("Extracting points");
    
    for i in 0..vertex_count {
        // Check for Python signals (Ctrl+C) every 10k points
        if i % 10000 == 0 {
            Python::with_gil(|py| -> PyResult<()> {
                py.check_signals()?;
                Ok(())
            })?;
        }
        
        let row_start = i * layout.row_size;
        let row_data = &vertex_data[row_start..row_start + layout.row_size];
        
        let x = f32::from_le_bytes([row_data[layout.x_offset], row_data[layout.x_offset+1], 
                                   row_data[layout.x_offset+2], row_data[layout.x_offset+3]]) as f64;
        let y = f32::from_le_bytes([row_data[layout.y_offset], row_data[layout.y_offset+1],
                                   row_data[layout.y_offset+2], row_data[layout.y_offset+3]]) as f64;
        let z = f32::from_le_bytes([row_data[layout.z_offset], row_data[layout.z_offset+1],
                                   row_data[layout.z_offset+2], row_data[layout.z_offset+3]]) as f64;
        
        // Apply Z filtering and sampling
        if z >= config.min_z && z < config.max_z {
            if config.sample_percentage >= 100.0 || 
               (config.sample_percentage > 0.0 && rng.gen::<f64>() < sample_threshold) {
                let r = layout.r_offset.map(|offset| row_data[offset]).unwrap_or(255);
                let g = layout.g_offset.map(|offset| row_data[offset]).unwrap_or(255);
                let b = layout.b_offset.map(|offset| row_data[offset]).unwrap_or(255);
                
                points.push(Point3D { x, y, z, r, g, b });
            }
        }
        
        // Update progress every 50k points
        if i % 50000 == 0 {
            pb.set_position(i as u64);
        }
    }
    
    pb.finish_with_message("Extraction complete");
    
    if config.sample_percentage < 100.0 {
        println!("Sampled {:.1}%: {} points", config.sample_percentage, points.len());
    }
    
    Ok(points)
}

pub fn assign_points_to_patches(points: &[Point3D], patches: &[Patch]) -> Result<Vec<Vec<usize>>, String> {
    let pb = ProgressBar::new(points.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})").unwrap());
    pb.set_message("Assigning to patches");
    
    let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); patches.len()];
    
    for (point_idx, point) in points.iter().enumerate() {
        // Check for Python signals (Ctrl+C) every 10k points
        if point_idx % 10000 == 0 {
            Python::with_gil(|py| -> PyResult<()> {
                py.check_signals()?;
                Ok(())
            }).map_err(|e| format!("Signal check failed: {}", e))?;
        }
        
        for (patch_idx, patch) in patches.iter().enumerate() {
            if point.x >= patch.min_x && point.x < patch.max_x && 
               point.y >= patch.min_y && point.y < patch.max_y {
                assignments[patch_idx].push(point_idx);
            }
        }
        
        // Update progress every 50k points
        if point_idx % 50000 == 0 {
            pb.set_position(point_idx as u64);
        }
    }
    
    pb.finish_with_message("Assignment complete");
    Ok(assignments)
}

pub fn write_colmap_file(output_path: &str, points: &[Point3D], indices: &[usize]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    
    // Write point count
    writer.write_all(&(indices.len() as u64).to_le_bytes())?;
    
    // Write points
    for (i, &point_idx) in indices.iter().enumerate() {
        let point = &points[point_idx];
        
        writer.write_all(&((i + 1) as u64).to_le_bytes())?;  // Point ID
        writer.write_all(&point.x.to_le_bytes())?;           // X
        writer.write_all(&point.y.to_le_bytes())?;           // Y
        writer.write_all(&point.z.to_le_bytes())?;           // Z
        writer.write_all(&[point.r, point.g, point.b])?;     // RGB
        writer.write_all(&(-1.0_f64).to_le_bytes())?;        // Error
        writer.write_all(&0_u64.to_le_bytes())?;             // Track length
    }
    
    writer.flush()?;
    Ok(())
}

#[pyfunction]
pub fn split_ply(config: &Config) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let results = PyDict::new(py);
        
        // Read PLY file
        let points = read_ply_file(&config.input_file, config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("PLY read error: {}", e)))?;
        
        results.set_item("points_loaded", points.len())?;
        
        // Assign points to patches
        let assignments = assign_points_to_patches(&points, &config.patches)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Assignment error: {}", e)))?;
        
        // Write output files
        let points_arc = Arc::new(points);
        let assignments_arc = Arc::new(assignments);
        
        // Progress bar for writing files
        let pb = ProgressBar::new(config.patches.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}").unwrap());
        pb.set_message("Writing patches");
        
        let mut handles = Vec::new();
        let mut patch_results = Vec::new();
        let patches_written = Arc::new(std::sync::Mutex::new(0));
        
        for (patch_idx, patch) in config.patches.iter().enumerate() {
            let points_clone = Arc::clone(&points_arc);
            let assignments_clone = Arc::clone(&assignments_arc);
            let written_counter = Arc::clone(&patches_written);
            let output_file = patch.output_file.clone();
            
            let handle = thread::spawn(move || -> Result<usize, String> {
                let indices = &assignments_clone[patch_idx];
                
                if !indices.is_empty() {
                    if let Err(e) = write_colmap_file(&output_file, &points_clone, indices) {
                        return Err(format!("Write error: {}", e));
                    }
                }
                
                // Update progress
                {
                    let mut count = written_counter.lock().unwrap();
                    *count += 1;
                }
                
                Ok(indices.len())
            });
            
            handles.push((handle, patch_idx));
        }
        
        // Monitor progress
        let mut last_written = 0;
        loop {
            // Check for Python signals (Ctrl+C)
            Python::with_gil(|py| -> PyResult<()> {
                py.check_signals()?;
                Ok(())
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyboardInterrupt, _>(format!("Interrupted: {}", e)))?;
            
            let current_written = *patches_written.lock().unwrap();
            if current_written > last_written {
                pb.set_position(current_written as u64);
                last_written = current_written;
            }
            if current_written >= config.patches.len() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        pb.finish_with_message("Writing complete");
        
        // Collect results
        let mut total_written = 0;
        for (handle, patch_idx) in handles {
            match handle.join() {
                Ok(Ok(count)) => {
                    total_written += count;
                    patch_results.push((patch_idx, count));
                }
                Ok(Err(e)) => return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e)),
                Err(_) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Thread join failed")),
            }
        }
        
        results.set_item("total_points_written", total_written)?;
        results.set_item("patches_written", patch_results.len())?;
        
        Ok(results.into())
    })
} 