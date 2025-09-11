use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

#[derive(Debug, Clone, Copy)]
struct PropertyLayout {
    x_offset: usize,
    y_offset: usize,
    z_offset: usize,
    vertex_size: usize,
}

#[pyclass]
pub struct OrthoCells {
    header_lines: Vec<String>,
    vertex_data: Vec<u8>,
    layout: PropertyLayout,
    vertex_count: usize,
}

#[pymethods]
impl OrthoCells {
    #[new]
    fn new(ply_path: String) -> PyResult<Self> {
        let mut file = File::open(&ply_path)?;
        let mut reader = BufReader::new(&mut file);

        let mut header_lines = Vec::new();
        let mut header_size = 0;
        let mut vertex_count = 0;
        let mut properties = Vec::new();

        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            header_size += bytes_read;
            let trimmed_line = line.trim();
            header_lines.push(trimmed_line.to_string());

            if trimmed_line == "end_header" {
                break;
            } else if trimmed_line.starts_with("element vertex ") {
                vertex_count = trimmed_line
                    .split_whitespace()
                    .nth(2)
                    .unwrap_or("0")
                    .parse()
                    .map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid vertex count")
                    })?;
            } else if trimmed_line.starts_with("property ") {
                let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
                if parts.len() >= 3 {
                    properties.push((parts[1].to_string(), parts[2].to_string()));
                }
            }
        }

        let mut layout = PropertyLayout {
            x_offset: 0,
            y_offset: 0,
            z_offset: 0,
            vertex_size: 0,
        };

        let mut offset = 0;
        for (prop_type, prop_name) in &properties {
            let size = match prop_type.as_str() {
                "float" | "float32" => 4,
                "double" | "float64" => 8,
                "uchar" | "uint8" => 1,
                "char" | "int8" => 1,
                "ushort" | "uint16" => 2,
                "short" | "int16" => 2,
                "uint" | "uint32" => 4,
                "int" | "int32" => 4,
                _ => 0,
            };

            match prop_name.as_str() {
                "x" => layout.x_offset = offset,
                "y" => layout.y_offset = offset,
                "z" => layout.z_offset = offset,
                _ => {}
            }
            offset += size;
        }
        layout.vertex_size = offset;

        file.seek(SeekFrom::Start(header_size as u64))?;

        let total_data_size = layout.vertex_size * vertex_count;
        let mut vertex_data = Vec::with_capacity(total_data_size);

        let pb = ProgressBar::new(total_data_size as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap(),
        );
        let file_name = std::path::Path::new(&ply_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("ply file");
        pb.set_message(format!("Reading {}", file_name));

        let mut reader_from_file = BufReader::new(file);
        let mut buffer = vec![0; 1024 * 1024];
        let mut read_bytes = 0;

        while read_bytes < total_data_size {
            let bytes_to_read = (total_data_size - read_bytes).min(buffer.len());
            let chunk = &mut buffer[..bytes_to_read];
            reader_from_file.read_exact(chunk)?;
            vertex_data.extend_from_slice(chunk);
            read_bytes += bytes_to_read;
            pb.set_position(read_bytes as u64);
        }

        pb.finish_with_message("PLY data loaded");

        Ok(OrthoCells {
            header_lines,
            vertex_data,
            layout,
            vertex_count,
        })
    }

    #[pyo3(text_signature = "($self, cells)")]
    fn extract_all_cells(&self, cells: &Bound<'_, PyList>) -> PyResult<()> {
        let mut patches: Vec<(f64, f64, f64, f64, String)> = Vec::with_capacity(cells.len());
        for cell_item in cells {
            let cell: &Bound<'_, PyTuple> = cell_item.downcast()?;
            let min_x: f64 = cell.get_item(0)?.extract()?;
            let max_x: f64 = cell.get_item(1)?.extract()?;
            let min_y: f64 = cell.get_item(2)?.extract()?;
            let max_y: f64 = cell.get_item(3)?.extract()?;
            let output_path: String = cell.get_item(4)?.extract()?;
            patches.push((min_x, max_x, min_y, max_y, output_path));
        }

        let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); patches.len()];

        let pb_assign = ProgressBar::new(self.vertex_count as u64);
        pb_assign.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
                .unwrap(),
        );
        pb_assign.set_message("Assigning splats to cells");

        for i in 0..self.vertex_count {
            let vertex_start = i * self.layout.vertex_size;
            let vertex_data =
                &self.vertex_data[vertex_start..vertex_start + self.layout.vertex_size];

            let x = f32::from_le_bytes([
                vertex_data[self.layout.x_offset],
                vertex_data[self.layout.x_offset + 1],
                vertex_data[self.layout.x_offset + 2],
                vertex_data[self.layout.x_offset + 3],
            ]) as f64;
            let y = f32::from_le_bytes([
                vertex_data[self.layout.y_offset],
                vertex_data[self.layout.y_offset + 1],
                vertex_data[self.layout.y_offset + 2],
                vertex_data[self.layout.y_offset + 3],
            ]) as f64;

            for (patch_idx, (min_x, max_x, min_y, max_y, _)) in patches.iter().enumerate() {
                if x >= *min_x && x <= *max_x && y >= *min_y && y <= *max_y {
                    assignments[patch_idx].push(i);
                }
            }
            if (i + 1) % 100000 == 0 {
                pb_assign.set_position((i + 1) as u64);
            }
        }
        pb_assign.finish_with_message("Assignment complete");

        let pb_write = ProgressBar::new(patches.len() as u64);
        pb_write.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
                .unwrap(),
        );
        pb_write.set_message("Writing cell PLY files");

        (0..patches.len()).into_par_iter().for_each(|patch_idx| {
            let indices = &assignments[patch_idx];
            if !indices.is_empty() {
                let (_, _, _, _, output_path) = &patches[patch_idx];
                let mut filtered_vertex_data =
                    Vec::with_capacity(indices.len() * self.layout.vertex_size);
                for &vertex_idx in indices {
                    let start = vertex_idx * self.layout.vertex_size;
                    let end = start + self.layout.vertex_size;
                    filtered_vertex_data.extend_from_slice(&self.vertex_data[start..end]);
                }
                if let Err(e) =
                    self.write_ply_file(output_path, &filtered_vertex_data, indices.len())
                {
                    eprintln!("Failed to write PLY file {}: {}", output_path, e);
                }
            }
            pb_write.inc(1);
        });
        pb_write.finish_with_message("Cell PLY files written");

        Ok(())
    }

    fn write_ply_file(
        &self,
        output_path: &str,
        vertex_data: &[u8],
        vertex_count: usize,
    ) -> PyResult<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        for line in &self.header_lines {
            if line.starts_with("element vertex ") {
                writeln!(writer, "element vertex {}", vertex_count)?;
            } else {
                writeln!(writer, "{}", line)?;
            }
        }

        writer.write_all(vertex_data)?;
        writer.flush()?;

        Ok(())
    }

    fn get_bounds(&self) -> PyResult<(f64, f64, f64, f64, f64)> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut sum_z = 0.0;

        for i in 0..self.vertex_count {
            let vertex_start = i * self.layout.vertex_size;
            let vertex_data =
                &self.vertex_data[vertex_start..vertex_start + self.layout.vertex_size];

            let x = f32::from_le_bytes([
                vertex_data[self.layout.x_offset],
                vertex_data[self.layout.x_offset + 1],
                vertex_data[self.layout.x_offset + 2],
                vertex_data[self.layout.x_offset + 3],
            ]) as f64;

            let y = f32::from_le_bytes([
                vertex_data[self.layout.y_offset],
                vertex_data[self.layout.y_offset + 1],
                vertex_data[self.layout.y_offset + 2],
                vertex_data[self.layout.y_offset + 3],
            ]) as f64;

            let z = f32::from_le_bytes([
                vertex_data[self.layout.z_offset],
                vertex_data[self.layout.z_offset + 1],
                vertex_data[self.layout.z_offset + 2],
                vertex_data[self.layout.z_offset + 3],
            ]) as f64;

            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            sum_z += z;
        }

        let avg_z = sum_z / self.vertex_count as f64;
        Ok((min_x, max_x, min_y, max_y, avg_z))
    }
}
