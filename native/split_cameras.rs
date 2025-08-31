use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SplitCamerasConfig {
    pub input_path: String,
    pub min_z: f64,
    pub max_z: f64,
    pub save_points3d: bool,
    pub patches: Vec<CameraPatchConfig>,
}

impl Default for SplitCamerasConfig {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            min_z: f64::NEG_INFINITY,
            max_z: f64::INFINITY,
            save_points3d: false,
            patches: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraPatchConfig {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub output_path: String,
}

impl Default for CameraPatchConfig {
    fn default() -> Self {
        Self {
            min_x: f64::NEG_INFINITY,
            min_y: f64::NEG_INFINITY,
            max_x: f64::INFINITY,
            max_y: f64::INFINITY,
            output_path: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub camera_id: u32,
    pub model_id: u32,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub image_id: u32,
    pub camera_id: u32,
    pub name: String,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub points2d: Vec<Point2D>,
}

impl Image {
    pub fn projection_center(&self) -> (f64, f64, f64) {
        let (qw, qx, qy, qz) = (self.qw, self.qx, self.qy, self.qz);
        let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
        let r01 = 2.0 * (qx * qy - qw * qz);
        let r02 = 2.0 * (qx * qz + qw * qy);
        let r10 = 2.0 * (qx * qy + qw * qz);
        let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
        let r12 = 2.0 * (qy * qz - qw * qx);
        let r20 = 2.0 * (qx * qz - qw * qy);
        let r21 = 2.0 * (qy * qz + qw * qx);
        let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
        (
            -(r00 * self.tx + r10 * self.ty + r20 * self.tz),
            -(r01 * self.tx + r11 * self.ty + r21 * self.tz),
            -(r02 * self.tx + r12 * self.ty + r22 * self.tz),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub point3d_id: u64,
}

#[derive(Debug, Clone)]
pub struct Point3D {
    pub point3d_id: u64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub error: f64,
    pub track: Vec<TrackElement>,
}

#[derive(Debug, Clone)]
pub struct TrackElement {
    pub image_id: u32,
    pub point2d_idx: u32,
}
pub struct ColmapSplitter {
    config: SplitCamerasConfig,
}

impl ColmapSplitter {
    pub fn new(config: SplitCamerasConfig) -> Self {
        ColmapSplitter { config }
    }

    pub fn split_cameras(&self) -> Result<SplitStats, Box<dyn std::error::Error>> {
        let cameras = self.read_cameras_binary()?;
        let original_images = self.read_images_binary()?;
        let original_points3d = self.read_points3d_binary()?;

        let results: Vec<Result<_, String>> = self
            .config
            .patches
            .par_iter()
            .map(|patch| {
                let kept_images: HashMap<u32, Image> = original_images
                    .iter()
                    .filter(|(_, img)| {
                        let (x, y, z) = img.projection_center();
                        patch.min_x <= x
                            && x <= patch.max_x
                            && patch.min_y <= y
                            && y <= patch.max_y
                            && self.config.min_z <= z
                            && z <= self.config.max_z
                    })
                    .map(|(id, img)| (*id, img.clone()))
                    .collect();
                let kept_points: HashMap<u64, Point3D> = original_points3d
                    .iter()
                    .filter(|(_, pt)| {
                        patch.min_x <= pt.x
                            && pt.x <= patch.max_x
                            && patch.min_y <= pt.y
                            && pt.y <= patch.max_y
                            && self.config.min_z <= pt.z
                            && pt.z <= self.config.max_z
                    })
                    .map(|(id, pt)| (*id, pt.clone()))
                    .collect();

                let used_camera_ids: std::collections::HashSet<_> =
                    kept_images.values().map(|img| img.camera_id).collect();
                let filtered_cameras: HashMap<u32, Camera> = cameras
                    .iter()
                    .filter(|(id, _)| used_camera_ids.contains(id))
                    .map(|(id, cam)| (*id, cam.clone()))
                    .collect();

                let (processed_images, idx_map) =
                    self.process_images_with_correspondences(&kept_images, &kept_points);
                let (final_points3d, point_id_mapping) =
                    self.process_points3d_with_tracks(&kept_points, &kept_images, &idx_map);
                let final_images =
                    self.update_images_with_new_point_ids(&processed_images, &point_id_mapping);

                let output_dir = Path::new(&patch.output_path);
                std::fs::create_dir_all(&output_dir).map_err(|e| e.to_string())?;

                // Write files directly to the output directory without creating sparse/0 subdirectory
                self.write_cameras_binary(&output_dir, &filtered_cameras)
                    .map_err(|e| e.to_string())?;
                self.write_images_binary(&output_dir, &final_images)
                    .map_err(|e| e.to_string())?;
                
                if self.config.save_points3d {
                    self.write_points3d_binary(&output_dir, &final_points3d)
                        .map_err(|e| e.to_string())?;
                }

                Ok((
                    1,
                    filtered_cameras.len(),
                    final_images.len(),
                    final_points3d.len(),
                ))
            })
            .collect();

        let patch_results: Result<Vec<_>, String> = results.into_iter().collect();
        let patch_results = patch_results.map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))
                as Box<dyn std::error::Error>
        })?;
        let (
            total_patches_written,
            total_cameras_written,
            total_images_written,
            total_points3d_written,
        ) = patch_results
            .iter()
            .fold((0, 0, 0, 0), |(p, c, i, pt), (dp, dc, di, dpt)| {
                (p + dp, c + dc, i + di, pt + dpt)
            });

        Ok(SplitStats {
            cameras_loaded: cameras.len(),
            images_loaded: original_images.len(),
            points3d_loaded: original_points3d.len(),
            patches_written: total_patches_written,
            total_cameras_written,
            total_images_written,
            total_points3d_written,
        })
    }

    fn process_images_with_correspondences(
        &self,
        kept_images: &HashMap<u32, Image>,
        kept_points: &HashMap<u64, Point3D>,
    ) -> (HashMap<u32, Image>, HashMap<u32, HashMap<u32, u32>>) {
        let mut processed_images = HashMap::new();
        let mut idx_map = HashMap::new();

        for (img_id, img) in kept_images {
            let (kept_points2d, img_idx_map): (Vec<_>, HashMap<_, _>) = img
                .points2d
                .iter()
                .enumerate()
                .filter(|(_, pt)| kept_points.contains_key(&pt.point3d_id))
                .enumerate()
                .map(|(new_i, (orig_i, pt))| (pt.clone(), (orig_i as u32, new_i as u32)))
                .unzip();

            idx_map.insert(*img_id, img_idx_map);
            processed_images.insert(
                *img_id,
                Image {
                    points2d: kept_points2d,
                    ..img.clone()
                },
            );
        }

        (processed_images, idx_map)
    }

    fn process_points3d_with_tracks(
        &self,
        kept_points: &HashMap<u64, Point3D>,
        kept_images: &HashMap<u32, Image>,
        idx_map: &HashMap<u32, HashMap<u32, u32>>,
    ) -> (HashMap<u64, Point3D>, HashMap<u64, u64>) {
        let mut final_points3d = HashMap::new();
        let mut new_point3d_id = 1u64;
        let mut point_id_mapping = HashMap::new();

        let mut sorted_point_ids: Vec<_> = kept_points.keys().collect();
        sorted_point_ids.sort_by_key(|k| std::cmp::Reverse(*k));
        for &original_pt_id in sorted_point_ids {
            let pt = &kept_points[&original_pt_id];
            let mut new_track: Vec<TrackElement> = pt
                .track
                .iter()
                .filter_map(|el| {
                    if kept_images.contains_key(&el.image_id) {
                        idx_map
                            .get(&el.image_id)?
                            .get(&el.point2d_idx)
                            .map(|&new_idx| TrackElement {
                                image_id: el.image_id,
                                point2d_idx: new_idx,
                            })
                    } else {
                        None
                    }
                })
                .collect();

            new_track.sort_by_key(|t| (t.image_id, t.point2d_idx));

            if !new_track.is_empty() {
                point_id_mapping.insert(original_pt_id, new_point3d_id);

                final_points3d.insert(
                    new_point3d_id,
                    Point3D {
                        point3d_id: new_point3d_id,
                        error: -1.0,
                        track: new_track,
                        ..pt.clone()
                    },
                );

                new_point3d_id += 1;
            }
        }

        (final_points3d, point_id_mapping)
    }

    fn update_images_with_new_point_ids(
        &self,
        images: &HashMap<u32, Image>,
        point_id_mapping: &HashMap<u64, u64>,
    ) -> HashMap<u32, Image> {
        images
            .iter()
            .map(|(img_id, image)| {
                let updated_points2d = image
                    .points2d
                    .iter()
                    .map(|pt| Point2D {
                        x: pt.x,
                        y: pt.y,
                        point3d_id: point_id_mapping
                            .get(&pt.point3d_id)
                            .copied()
                            .unwrap_or(pt.point3d_id),
                    })
                    .collect();

                (
                    *img_id,
                    Image {
                        points2d: updated_points2d,
                        ..image.clone()
                    },
                )
            })
            .collect()
    }

    fn read_cameras_binary(&self) -> Result<HashMap<u32, Camera>, Box<dyn std::error::Error>> {
        let camera_path = Path::new(&self.config.input_path).join("cameras.bin");
        let file = File::open(camera_path)?;
        let mut reader = BufReader::new(file);

        let num_cameras = reader.read_u64::<LittleEndian>()?;
        let mut cameras = HashMap::new();

        for _ in 0..num_cameras {
            let camera_id = reader.read_u32::<LittleEndian>()?;
            let model_id = reader.read_u32::<LittleEndian>()?;
            let width = reader.read_u64::<LittleEndian>()?;
            let height = reader.read_u64::<LittleEndian>()?;

            let num_params = match model_id {
                0 => 3,
                1 => 4,
                2 => 5,
                3 => 8,
                4 => 8,
                5 => 12,
                _ => return Err(format!("Unknown camera model: {}", model_id).into()),
            };

            let mut params = Vec::with_capacity(num_params);
            for _ in 0..num_params {
                params.push(reader.read_f64::<LittleEndian>()?);
            }

            cameras.insert(
                camera_id,
                Camera {
                    camera_id,
                    model_id,
                    width,
                    height,
                    params,
                },
            );
        }

        Ok(cameras)
    }

    fn read_images_binary(&self) -> Result<HashMap<u32, Image>, Box<dyn std::error::Error>> {
        let images_path = Path::new(&self.config.input_path).join("images.bin");
        let file = File::open(images_path)?;
        let mut reader = BufReader::new(file);

        let num_images = reader.read_u64::<LittleEndian>()?;
        let mut images = HashMap::new();

        for _ in 0..num_images {
            let image_id = reader.read_u32::<LittleEndian>()?;

            let (qw, qx, qy, qz) = (
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
            );
            let (tx, ty, tz) = (
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
            );

            let camera_id = reader.read_u32::<LittleEndian>()?;

            let mut name_bytes = Vec::new();
            loop {
                let byte = reader.read_u8()?;
                if byte == 0 {
                    break;
                }
                name_bytes.push(byte);
            }
            let name = String::from_utf8(name_bytes)?;

            let num_points2d = reader.read_u64::<LittleEndian>()?;
            let mut points2d = Vec::with_capacity(num_points2d as usize);

            for _ in 0..num_points2d {
                points2d.push(Point2D {
                    x: reader.read_f64::<LittleEndian>()?,
                    y: reader.read_f64::<LittleEndian>()?,
                    point3d_id: reader.read_u64::<LittleEndian>()?,
                });
            }

            images.insert(
                image_id,
                Image {
                    image_id,
                    camera_id,
                    name,
                    qw,
                    qx,
                    qy,
                    qz,
                    tx,
                    ty,
                    tz,
                    points2d,
                },
            );
        }

        Ok(images)
    }

    fn read_points3d_binary(&self) -> Result<HashMap<u64, Point3D>, Box<dyn std::error::Error>> {
        let points3d_path = Path::new(&self.config.input_path).join("points3D.bin");
        let file = File::open(points3d_path)?;
        let mut reader = BufReader::new(file);

        let num_points3d = reader.read_u64::<LittleEndian>()?;
        let mut points3d = HashMap::with_capacity(num_points3d as usize);

        for _ in 0..num_points3d {
            let point3d_id = reader.read_u64::<LittleEndian>()?;
            let (x, y, z) = (
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
                reader.read_f64::<LittleEndian>()?,
            );
            let (r, g, b) = (reader.read_u8()?, reader.read_u8()?, reader.read_u8()?);
            let error = reader.read_f64::<LittleEndian>()?;
            let track_length = reader.read_u64::<LittleEndian>()?;

            let mut track = Vec::with_capacity(track_length as usize);
            for _ in 0..track_length {
                track.push(TrackElement {
                    image_id: reader.read_u32::<LittleEndian>()?,
                    point2d_idx: reader.read_u32::<LittleEndian>()?,
                });
            }

            points3d.insert(
                point3d_id,
                Point3D {
                    point3d_id,
                    x,
                    y,
                    z,
                    r,
                    g,
                    b,
                    error,
                    track,
                },
            );
        }

        Ok(points3d)
    }

    fn write_cameras_binary(
        &self,
        output_dir: &Path,
        cameras: &HashMap<u32, Camera>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;
        let camera_path = output_dir.join("cameras.bin");
        let file = File::create(camera_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_u64::<LittleEndian>(cameras.len() as u64)?;
        let mut sorted: Vec<_> = cameras.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        for (_, camera) in sorted {
            writer.write_u32::<LittleEndian>(camera.camera_id)?;
            writer.write_u32::<LittleEndian>(camera.model_id)?;
            writer.write_u64::<LittleEndian>(camera.width)?;
            writer.write_u64::<LittleEndian>(camera.height)?;
            for &param in &camera.params {
                writer.write_f64::<LittleEndian>(param)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    fn write_images_binary(
        &self,
        output_dir: &Path,
        images: &HashMap<u32, Image>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;
        let images_path = output_dir.join("images.bin");
        let file = File::create(images_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_u64::<LittleEndian>(images.len() as u64)?;
        let mut sorted: Vec<_> = images.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        for (_, image) in sorted {
            writer.write_u32::<LittleEndian>(image.image_id)?;
            for &val in &[
                image.qw, image.qx, image.qy, image.qz, image.tx, image.ty, image.tz,
            ] {
                writer.write_f64::<LittleEndian>(val)?;
            }
            writer.write_u32::<LittleEndian>(image.camera_id)?;

            writer.write_all(image.name.as_bytes())?;
            writer.write_u8(0)?;

            writer.write_u64::<LittleEndian>(image.points2d.len() as u64)?;
            for point2d in &image.points2d {
                writer.write_f64::<LittleEndian>(point2d.x)?;
                writer.write_f64::<LittleEndian>(point2d.y)?;
                writer.write_u64::<LittleEndian>(point2d.point3d_id)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    fn write_points3d_binary(
        &self,
        output_dir: &Path,
        points3d: &HashMap<u64, Point3D>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(output_dir)?;
        let points3d_path = output_dir.join("points3D.bin");
        let file = File::create(points3d_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_u64::<LittleEndian>(points3d.len() as u64)?;
        let mut sorted: Vec<_> = points3d.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        for (_, pt) in sorted {
            writer.write_u64::<LittleEndian>(pt.point3d_id)?;
            for &val in &[pt.x, pt.y, pt.z] {
                writer.write_f64::<LittleEndian>(val)?;
            }
            for &val in &[pt.r, pt.g, pt.b] {
                writer.write_u8(val)?;
            }
            writer.write_f64::<LittleEndian>(pt.error)?;

            writer.write_u64::<LittleEndian>(pt.track.len() as u64)?;
            for el in &pt.track {
                writer.write_u32::<LittleEndian>(el.image_id)?;
                writer.write_u32::<LittleEndian>(el.point2d_idx)?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct SplitStats {
    pub cameras_loaded: usize,
    pub images_loaded: usize,
    pub points3d_loaded: usize,
    pub patches_written: usize,
    pub total_cameras_written: usize,
    pub total_images_written: usize,
    pub total_points3d_written: usize,
}

#[pyfunction]
pub fn split_cameras_json(config_json: &str) -> PyResult<HashMap<String, usize>> {
    let config: SplitCamerasConfig = serde_json::from_str(config_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e)))?;
    
    let stats = ColmapSplitter::new(config).split_cameras().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Camera splitting failed: {}", e))
    })?;

    Ok([
        ("cameras_loaded", stats.cameras_loaded),
        ("images_loaded", stats.images_loaded),
        ("points3d_loaded", stats.points3d_loaded),
        ("patches_written", stats.patches_written),
        ("total_cameras_written", stats.total_cameras_written),
        ("total_images_written", stats.total_images_written),
        ("total_points3d_written", stats.total_points3d_written),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), *v))
    .collect())
}
