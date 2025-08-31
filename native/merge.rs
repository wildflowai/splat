/// Concatenates multiple PLY files into one file.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

/// Configuration for merging multiple PLY files into one.
#[derive(Debug, Deserialize, Serialize, Clone)]
struct MergeConfig {
    pub input_files: Vec<String>,
    pub output_file: String,
}

/// PLY file header information
#[derive(Debug, Clone)]
struct PlyHeader {
    pub vertex_count: usize,
    pub header_size: usize,
    pub header_text: String,
    pub row_size: usize,
}

/// Read PLY header and return header info
fn read_ply_header(
    reader: &mut BufReader<&mut File>,
) -> Result<PlyHeader, Box<dyn std::error::Error>> {
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

    // Calculate row size based on properties
    let mut row_size = 0;
    for (prop_type, _prop_name) in &properties {
        let size = match prop_type.as_str() {
            "float" | "f32" => 4,
            "double" | "f64" => 8,
            "uchar" | "u8" => 1,
            "int" | "i32" => 4,
            _ => 4,
        };
        row_size += size;
    }

    Ok(PlyHeader {
        vertex_count,
        header_size,
        header_text: header,
        row_size,
    })
}

/// Merge multiple PLY files into one
fn merge_ply_files(config: &MergeConfig) -> PyResult<()> {
    let start_time = Instant::now();
    println!("Starting PLY merge process...");

    if config.input_files.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No input files provided for merging.",
        ));
    }

    // First pass: read headers and count total vertices
    let mut total_vertices = 0;
    let mut headers = Vec::new();
    let mut file_sizes = Vec::new();

    println!("Reading PLY headers and counting vertices...");
    for input_file in &config.input_files {
        let mut file = File::open(input_file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut reader = BufReader::new(&mut file);

        let header = read_ply_header(&mut reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        total_vertices += header.vertex_count;
        headers.push(header.clone());
        file_sizes.push(file.metadata()?.len());

        println!("  {}: {} vertices", input_file, header.vertex_count);
    }

    println!("Total vertices across all files: {}", total_vertices);

    // Create output file
    let output_file = File::create(&config.output_file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let mut writer = BufWriter::new(output_file);

    // Write modified header with total vertex count
    let first_header = &headers[0];
    let new_header = first_header.header_text.replace(
        &format!("element vertex {}", first_header.vertex_count),
        &format!("element vertex {}", total_vertices),
    );
    writer.write_all(new_header.as_bytes())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Second pass: concatenate vertex data from all files
    println!("Concatenating vertex data...");
    for (i, input_file) in config.input_files.iter().enumerate() {
        println!("  Reading vertex data from {}...", input_file);
        let read_start = Instant::now();

        let mut file = File::open(input_file)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        // Skip header
        file.seek(SeekFrom::Start(headers[i].header_size as u64))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // Read and write vertex data
        let mut buffer = vec![0u8; 8192]; // 8KB buffer
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            if bytes_read == 0 {
                break;
            }
            writer.write_all(&buffer[..bytes_read])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        }

        println!("    Added {} vertices in {:?}", headers[i].vertex_count, read_start.elapsed());
    }

    // Flush writer
    writer.flush()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    println!(
        "PLY merge process completed in {:?}",
        start_time.elapsed()
    );
    println!("Successfully merged {} files into {}", config.input_files.len(), config.output_file);
    println!("Total vertices: {}", total_vertices);

    Ok(())
}

#[pyfunction]
pub fn merge_ply_files_py(config_json: &str) -> PyResult<()> {
    let config: MergeConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    merge_ply_files(&config)
}
