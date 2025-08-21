use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::collections::HashMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// Default value functions for serde
fn default_neg_inf() -> f64 { f64::NEG_INFINITY }
fn default_pos_inf() -> f64 { f64::INFINITY }

/// Configuration for COLMAP camera splitting
#[derive(Debug, Deserialize, Serialize, Clone)]
#[pyclass]
pub struct CameraConfig {
    #[pyo3(get, set)]
    pub input_path: String,
    #[pyo3(get, set)]
    #[serde(rename = "minZ", default = "default_neg_inf")]
    pub min_z: f64,
    #[pyo3(get, set)]
    #[serde(rename = "maxZ", default = "default_pos_inf")]
    pub max_z: f64,
    #[pyo3(get, set)]
    pub patches: Vec<CameraPatch>,
}

#[pymethods]
impl CameraConfig {
    #[new]
    fn new(input_path: String) -> Self {
        CameraConfig {
            input_path,
            min_z: f64::NEG_INFINITY,
            max_z: f64::INFINITY,
            patches: Vec::new(),
        }
    }

    fn add_patch(&mut self, patch: CameraPatch) {
        self.patches.push(patch);
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<CameraConfig> {
        serde_json::from_str(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON parse error: {}", e)))
    }

    #[staticmethod]
    fn from_file(file_path: &str) -> PyResult<CameraConfig> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File read error: {}", e)))?;
        Self::from_json(&content)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialize error: {}", e)))
    }
}

/// Camera patch definition with spatial bounds
#[derive(Debug, Clone, Deserialize, Serialize)]
#[pyclass]
pub struct CameraPatch {
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
    pub output_path: String,
}

#[pymethods]
impl CameraPatch {
    #[new]
    fn new(output_path: String) -> Self {
        CameraPatch {
            min_x: f64::NEG_INFINITY,
            min_y: f64::NEG_INFINITY,
            max_x: f64::INFINITY,
            max_y: f64::INFINITY,
            output_path,
        }
    }

    fn set_bounds(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        self.min_x = min_x;
        self.min_y = min_y;
        self.max_x = max_x;
        self.max_y = max_y;
    }

    fn contains_camera_position(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

/// COLMAP camera structure
#[derive(Debug, Clone)]
pub struct Camera {
    pub camera_id: u32,
    pub model_id: u32,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

/// COLMAP image structure
#[derive(Debug, Clone)]  
pub struct Image {
    pub image_id: u32,
    pub camera_id: u32,
    pub name: String,
    pub qw: f64, pub qx: f64, pub qy: f64, pub qz: f64, // Quaternion rotation
    pub tx: f64, pub ty: f64, pub tz: f64,             // Translation
    pub points2d: Vec<Point2D>,
}

impl Image {
    /// Calculate camera projection center from pose
    pub fn projection_center(&self) -> (f64, f64, f64) {
        // Convert quaternion to rotation matrix and compute camera center
        // C = -R^T * t where R is rotation matrix from quaternion
        let qw = self.qw; let qx = self.qx; let qy = self.qy; let qz = self.qz;
        
        // Rotation matrix from quaternion (standard formula - matching COLMAP/Eigen)
        let r00 = 1.0 - 2.0 * (qy*qy + qz*qz);
        let r01 = 2.0 * (qx*qy - qw*qz);
        let r02 = 2.0 * (qx*qz + qw*qy);
        let r10 = 2.0 * (qx*qy + qw*qz);
        let r11 = 1.0 - 2.0 * (qx*qx + qz*qz);
        let r12 = 2.0 * (qy*qz - qw*qx);
        let r20 = 2.0 * (qx*qz - qw*qy);
        let r21 = 2.0 * (qy*qz + qw*qx);
        let r22 = 1.0 - 2.0 * (qx*qx + qy*qy);
        
        // Camera center: C = -R^T * t
        let cx = -(r00 * self.tx + r10 * self.ty + r20 * self.tz);
        let cy = -(r01 * self.tx + r11 * self.ty + r21 * self.tz);
        let cz = -(r02 * self.tx + r12 * self.ty + r22 * self.tz);
        
        (cx, cy, cz)
    }
}

/// COLMAP 2D point structure
#[derive(Debug, Clone)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub point3d_id: u64, // INVALID_POINT3D_ID = 18446744073709551615 if no 3D point
}

/// COLMAP 3D point structure
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

impl Point3D {
    pub fn position(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
    
    pub fn in_bounds(&self, min_z: f64, max_z: f64) -> bool {
        self.z >= min_z && self.z <= max_z
    }
}

/// Track element linking 2D and 3D points
#[derive(Debug, Clone)]
pub struct TrackElement {
    pub image_id: u32,
    pub point2d_idx: u32,
}

/// Main COLMAP reconstruction splitter
pub struct ColmapSplitter {
    config: CameraConfig,
}

impl ColmapSplitter {
    pub fn new(config: CameraConfig) -> Self {
        ColmapSplitter { config }
    }
    
    /// Split COLMAP reconstruction into multiple patches
    pub fn split_cameras(&self) -> Result<SplitStats, Box<dyn std::error::Error>> {
        // Load COLMAP data once
        let cameras = self.read_cameras_binary()?;
        let original_images = self.read_images_binary()?;
        let original_points3d = self.read_points3d_binary()?;
        
        let mut total_patches_written = 0;
        let mut total_cameras_written = 0;
        let mut total_images_written = 0;
        let mut total_points3d_written = 0;
        
        // Process each patch
        for patch in &self.config.patches {
            // STEP 1: Filter images by camera position (exact Python logic)
            let kept_images = self.filter_images_by_camera_position(&original_images, patch);
            
            // STEP 2: Filter 3D points by position (exact Python logic)  
            let kept_points = self.filter_points3d_by_position(&original_points3d, patch);
            
            // STEP 3: Add relevant cameras (exact Python logic)
            let filtered_cameras = self.get_cameras_for_images(&cameras, &kept_images);
            
            // STEP 4: Process images with filtered 2D-3D correspondences (exact Python logic)
            let (processed_images, idx_map) = self.process_images_with_correspondences(&kept_images, &kept_points);
            
            // STEP 5: Process 3D points with updated tracks (exact Python logic)
            let (final_points3d, point_id_mapping) = self.process_points3d_with_tracks(&kept_points, &kept_images, &idx_map);
            
            // STEP 6: Update 2D points to reference new sequential point3D IDs
            let final_images = self.update_images_with_new_point_ids(&processed_images, &point_id_mapping);
            
            // Create output directory
            let output_dir = Path::new(&patch.output_path).join("sparse").join("0");
            std::fs::create_dir_all(&output_dir)?;
            
                        // Write filtered data for this patch
        self.write_cameras_binary(&output_dir, &filtered_cameras)?;
            self.write_images_binary(&output_dir, &final_images)?;
        self.write_points3d_binary(&output_dir, &final_points3d)?;
        
            total_patches_written += 1;
            total_cameras_written += filtered_cameras.len();
            total_images_written += final_images.len();
            total_points3d_written += final_points3d.len();
        }
        
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
    
    // STEP 1: Filter images by camera position (Python: kept_images = {img_id: img for img_id, img in model.images.items() if ...})
    fn filter_images_by_camera_position(&self, images: &HashMap<u32, Image>, patch: &CameraPatch) -> HashMap<u32, Image> {
        let mut kept_images = HashMap::new();
        
        for (img_id, img) in images {
            let pos = img.projection_center();
            
            // Exact Python logic: (min_x <= (pos := img.projection_center())[0] <= max_x and min_y <= pos[1] <= max_y and min_z <= pos[2] <= max_z)
            let x_check = patch.min_x <= pos.0 && pos.0 <= patch.max_x;
            let y_check = patch.min_y <= pos.1 && pos.1 <= patch.max_y;
            let z_check = self.config.min_z <= pos.2 && pos.2 <= self.config.max_z;
            

            
            if x_check && y_check && z_check {
                kept_images.insert(*img_id, img.clone());
            }
        }
        
        kept_images
    }
    
    // STEP 2: Filter 3D points by position (Python: kept_points = {pt_id: pt for pt_id, pt in model.points3D.items() if ...})
    fn filter_points3d_by_position(&self, points3d: &HashMap<u64, Point3D>, patch: &CameraPatch) -> HashMap<u64, Point3D> {
        let mut kept_points = HashMap::new();
        
        for (pt_id, pt) in points3d {
            let xyz = pt.position();
            // Exact Python logic: (min_x <= (xyz := pt.xyz)[0] <= max_x and min_y <= xyz[1] <= max_y and min_z <= xyz[2] <= max_z)
            if patch.min_x <= xyz.0 && xyz.0 <= patch.max_x &&
               patch.min_y <= xyz.1 && xyz.1 <= patch.max_y &&
               self.config.min_z <= xyz.2 && xyz.2 <= self.config.max_z {
                kept_points.insert(*pt_id, pt.clone());
            }
        }
        
        kept_points
    }
    
    // STEP 3: Add relevant cameras (Python: for camera_id in {img.camera_id for img in kept_images.values()}: new_model.add_camera(model.cameras[camera_id]))
    fn get_cameras_for_images(&self, cameras: &HashMap<u32, Camera>, kept_images: &HashMap<u32, Image>) -> HashMap<u32, Camera> {
        let mut filtered_cameras = HashMap::new();
        
        // Get unique camera IDs from kept images
        let mut used_camera_ids = std::collections::HashSet::new();
        for img in kept_images.values() {
            used_camera_ids.insert(img.camera_id);
        }
        
        // Add cameras for those IDs
        for camera_id in used_camera_ids {
            if let Some(camera) = cameras.get(&camera_id) {
                filtered_cameras.insert(camera_id, camera.clone());
            }
        }
        
        filtered_cameras
    }
    
    // STEP 4: Process images with filtered 2D-3D correspondences (Python: for img_id, img in tqdm(kept_images.items(), desc="Processing images"):)
    fn process_images_with_correspondences(&self, kept_images: &HashMap<u32, Image>, kept_points: &HashMap<u64, Point3D>) -> (HashMap<u32, Image>, HashMap<u32, HashMap<u32, u32>>) {
        const INVALID_POINT3D_ID: u64 = 18446744073709551615;
        let mut processed_images = HashMap::new();
        let mut idx_map = HashMap::new();
        
        for (img_id, img) in kept_images {
            // Python: points2D = img.points2D
            let points2d = &img.points2d;
            
            // Python: kept_points2D = [pycolmap.Point2D(pt.xy.reshape(2, 1), invalid_id) for pt in points2D if pt.point3D_id in kept_points]
            // Note: Python sets invalid_id initially, but we'll preserve the actual point3D_id since we handle linking differently
            let mut kept_points2d = Vec::new();
            for pt in points2d {
                if kept_points.contains_key(&pt.point3d_id) {
                    kept_points2d.push(Point2D {
                        x: pt.x,
                        y: pt.y,
                        point3d_id: pt.point3d_id, // Keep the actual 3D point ID
                    });
                }
            }
            
            // Python: original_indices = [i for i, pt in enumerate(points2D) if pt.point3D_id in kept_points]
            let mut original_indices = Vec::new();
            for (i, pt) in points2d.iter().enumerate() {
                if kept_points.contains_key(&pt.point3d_id) {
                    original_indices.push(i as u32);
                }
            }
            
            // Python: idx_map[img_id] = {orig_i: new_i for new_i, orig_i in enumerate(original_indices)}
            let mut img_idx_map = HashMap::new();
            for (new_i, orig_i) in original_indices.iter().enumerate() {
                img_idx_map.insert(*orig_i, new_i as u32);
            }
            idx_map.insert(*img_id, img_idx_map);
            
            // Python always adds the image regardless of whether kept_points2D is empty (lines 72-79)
            // Python: new_model.add_image(new_img) is called unconditionally
            processed_images.insert(*img_id, Image {
                image_id: img.image_id,
                camera_id: img.camera_id,
                name: img.name.clone(),
                qw: img.qw, qx: img.qx, qy: img.qy, qz: img.qz,
                tx: img.tx, ty: img.ty, tz: img.tz,
                points2d: kept_points2d,
            });
        }
        
        (processed_images, idx_map)
    }
    
    // STEP 5: Process 3D points with updated tracks (Python: for pt_id, pt in tqdm(kept_points.items(), desc="Processing 3D points"):)
    fn process_points3d_with_tracks(&self, kept_points: &HashMap<u64, Point3D>, kept_images: &HashMap<u32, Image>, idx_map: &HashMap<u32, HashMap<u32, u32>>) -> (HashMap<u64, Point3D>, HashMap<u64, u64>) {
        let mut final_points3d = HashMap::new();
        
        // Create sequential point3D ID mapping (like Python add_point3D)
        let mut new_point3d_id = 1u64;
        let mut point_id_mapping = HashMap::new();
        
        // Sort original point IDs for consistent processing order (reverse order to match Python)
        let mut sorted_point_ids: Vec<_> = kept_points.keys().collect();
        sorted_point_ids.sort();
        sorted_point_ids.reverse();
        
        for &original_pt_id in sorted_point_ids {
            let pt = &kept_points[&original_pt_id];
            // Python: track = pycolmap.Track()
            let mut new_track = Vec::new();
            
            // Python: for el in pt.track.elements: if el.image_id in kept_images and (new_idx := idx_map[el.image_id].get(el.point2D_idx)) is not None:
            for track_element in &pt.track {
                if kept_images.contains_key(&track_element.image_id) {
                    if let Some(img_idx_map) = idx_map.get(&track_element.image_id) {
                        if let Some(&new_point2d_idx) = img_idx_map.get(&track_element.point2d_idx) {
                            // Python: track.add_element(el.image_id, new_idx)
                            new_track.push(TrackElement {
                                image_id: track_element.image_id,
                                point2d_idx: new_point2d_idx,
                            });
                        }
                    }
                }
            }
            
            // Sort track elements by (image_id, point2d_idx) for consistent order
            new_track.sort_by_key(|t| (t.image_id, t.point2d_idx));
            
            // Python: if track.elements: new_model.add_point3D(pt.xyz, track, color=pt.color)
            if !new_track.is_empty() {
                // Assign new sequential ID (like Python add_point3D)
                point_id_mapping.insert(original_pt_id, new_point3d_id);
                
                final_points3d.insert(new_point3d_id, Point3D {
                    point3d_id: new_point3d_id,
                    x: pt.x, y: pt.y, z: pt.z,
                    r: pt.r, g: pt.g, b: pt.b,
                    error: -1.0, // Match Python add_point3D behavior
                    track: new_track,
                });
                
                new_point3d_id += 1;
            }
        }
        
        (final_points3d, point_id_mapping)
    }
    
    // STEP 6: Update images to use new sequential point3D IDs
    fn update_images_with_new_point_ids(&self, images: &HashMap<u32, Image>, point_id_mapping: &HashMap<u64, u64>) -> HashMap<u32, Image> {
        let mut final_images = HashMap::new();
        
        for (img_id, image) in images {
            let mut updated_points2d = Vec::new();
            
            for point2d in &image.points2d {
                let new_point3d_id = point_id_mapping.get(&point2d.point3d_id)
                    .copied()
                    .unwrap_or(point2d.point3d_id); // Keep original if not in mapping (shouldn't happen)
                
                updated_points2d.push(Point2D {
                    x: point2d.x,
                    y: point2d.y,
                    point3d_id: new_point3d_id,
                });
            }
            
            final_images.insert(*img_id, Image {
                image_id: image.image_id,
                camera_id: image.camera_id,
                name: image.name.clone(),
                qw: image.qw, qx: image.qx, qy: image.qy, qz: image.qz,
                tx: image.tx, ty: image.ty, tz: image.tz,
                points2d: updated_points2d,
            });
        }
        
        final_images
    }
    
    fn read_cameras_binary(&self) -> Result<HashMap<u32, Camera>, Box<dyn std::error::Error>> {
        let camera_path = Path::new(&self.config.input_path).join("cameras.bin");
        let mut file = File::open(camera_path)?;
        let mut reader = BufReader::new(file);
        
        // Read number of cameras
        let num_cameras = reader.read_u64::<LittleEndian>()?;
        let mut cameras = HashMap::new();
        
        for _ in 0..num_cameras {
            // Read camera data
            let camera_id = reader.read_u32::<LittleEndian>()?;
            let model_id = reader.read_u32::<LittleEndian>()?;
            let width = reader.read_u64::<LittleEndian>()?;
            let height = reader.read_u64::<LittleEndian>()?;
            
            // Read parameters based on model
            let num_params = match model_id {
                0 => 3,  // SIMPLE_PINHOLE: f, cx, cy
                1 => 4,  // PINHOLE: fx, fy, cx, cy
                2 => 5,  // SIMPLE_RADIAL: f, cx, cy, k
                3 => 8,  // RADIAL: f, cx, cy, k1, k2, k3, k4
                4 => 8,  // OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                5 => 12, // OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, p3, p4
                _ => return Err(format!("Unknown camera model: {}", model_id).into()),
            };
            
            let mut params = Vec::with_capacity(num_params);
            for _ in 0..num_params {
                params.push(reader.read_f64::<LittleEndian>()?);
            }
            
            cameras.insert(camera_id, Camera {
                camera_id,
                model_id,
                width,
                height,
                params,
            });
        }
        
        Ok(cameras)
    }
    
    fn read_images_binary(&self) -> Result<HashMap<u32, Image>, Box<dyn std::error::Error>> {
        let images_path = Path::new(&self.config.input_path).join("images.bin");
        let file = File::open(images_path)?;
        let mut reader = BufReader::new(file);
        
        // Read number of images
        let num_images = reader.read_u64::<LittleEndian>()?;
        let mut images = HashMap::new();
        
        for _ in 0..num_images {
            // Read image metadata
            let image_id = reader.read_u32::<LittleEndian>()?;
            
            // Read quaternion (w, x, y, z)
            let qw = reader.read_f64::<LittleEndian>()?;
            let qx = reader.read_f64::<LittleEndian>()?;
            let qy = reader.read_f64::<LittleEndian>()?;
            let qz = reader.read_f64::<LittleEndian>()?;
            
            // Read translation (tx, ty, tz)
            let tx = reader.read_f64::<LittleEndian>()?;
            let ty = reader.read_f64::<LittleEndian>()?;
            let tz = reader.read_f64::<LittleEndian>()?;
            
            // Read camera ID
            let camera_id = reader.read_u32::<LittleEndian>()?;
            
            // Read image name (null-terminated string)
            let mut name_bytes = Vec::new();
            loop {
                let byte = reader.read_u8()?;
                if byte == 0 { break; }
                name_bytes.push(byte);
            }
            let name = String::from_utf8(name_bytes)?;
            
            // Read 2D points
            let num_points2d = reader.read_u64::<LittleEndian>()?;
            let mut points2d = Vec::with_capacity(num_points2d as usize);
            
            for _ in 0..num_points2d {
                let x = reader.read_f64::<LittleEndian>()?;
                let y = reader.read_f64::<LittleEndian>()?;
                let point3d_id = reader.read_u64::<LittleEndian>()?;
                
                points2d.push(Point2D { x, y, point3d_id });
            }
            
            images.insert(image_id, Image {
                image_id,
                camera_id,
                name,
                qw, qx, qy, qz,
                tx, ty, tz,
                points2d,
            });
        }
        
        Ok(images)
    }
    
    fn read_points3d_binary(&self) -> Result<HashMap<u64, Point3D>, Box<dyn std::error::Error>> {
        let points3d_path = Path::new(&self.config.input_path).join("points3D.bin");
        let file = File::open(points3d_path)?;
        let mut reader = BufReader::new(file);
        
        // Read number of 3D points
        let num_points3d = reader.read_u64::<LittleEndian>()?;
        let mut points3d = HashMap::new();
        
        for _ in 0..num_points3d {
            // Read point3D ID
            let point3d_id = reader.read_u64::<LittleEndian>()?;
            
            // Read 3D coordinates
            let x = reader.read_f64::<LittleEndian>()?;
            let y = reader.read_f64::<LittleEndian>()?;
            let z = reader.read_f64::<LittleEndian>()?;
            
            // Read color (RGB)
            let r = reader.read_u8()?;
            let g = reader.read_u8()?;
            let b = reader.read_u8()?;
            
            // Read error
            let error = reader.read_f64::<LittleEndian>()?;
            
            // Read track (list of image_id, point2D_idx pairs)
            let track_length = reader.read_u64::<LittleEndian>()?;
            let mut track = Vec::with_capacity(track_length as usize);
            
            for _ in 0..track_length {
                let image_id = reader.read_u32::<LittleEndian>()?;
                let point2d_idx = reader.read_u32::<LittleEndian>()?;
                track.push(TrackElement { image_id, point2d_idx });
            }
            
            points3d.insert(point3d_id, Point3D {
                point3d_id,
                x, y, z,
                r, g, b,
                error,
                track,
            });
        }
        
        Ok(points3d)
    }
    
    fn write_cameras_binary(&self, output_dir: &Path, cameras: &HashMap<u32, Camera>) -> Result<(), Box<dyn std::error::Error>> {
        let camera_path = output_dir.join("cameras.bin");
        let file = File::create(camera_path)?;
        let mut writer = BufWriter::new(file);
        
        // Write number of cameras
        writer.write_u64::<LittleEndian>(cameras.len() as u64)?;
        
        // Sort cameras by ID for consistent output order (matching COLMAP reference)
        let mut sorted_cameras: Vec<_> = cameras.iter().collect();
        sorted_cameras.sort_by_key(|(camera_id, _)| *camera_id);
        
        for (_, camera) in sorted_cameras {
            // Write camera data
            writer.write_u32::<LittleEndian>(camera.camera_id)?;
            writer.write_u32::<LittleEndian>(camera.model_id)?;
            writer.write_u64::<LittleEndian>(camera.width)?;
            writer.write_u64::<LittleEndian>(camera.height)?;
            
            // Write parameters
            for &param in &camera.params {
                writer.write_f64::<LittleEndian>(param)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    fn write_images_binary(&self, output_dir: &Path, images: &HashMap<u32, Image>) -> Result<(), Box<dyn std::error::Error>> {
        let images_path = output_dir.join("images.bin");
        let file = File::create(images_path)?;
        let mut writer = BufWriter::new(file);
        
        // Write number of images
        writer.write_u64::<LittleEndian>(images.len() as u64)?;
        
        // Sort images by ID for consistent output order (matching COLMAP reference)
        let mut sorted_images: Vec<_> = images.iter().collect();
        sorted_images.sort_by_key(|(image_id, _)| *image_id);
        
        for (_, image) in sorted_images {
            // Write image metadata
            writer.write_u32::<LittleEndian>(image.image_id)?;
            
            // Write quaternion (w, x, y, z)
            writer.write_f64::<LittleEndian>(image.qw)?;
            writer.write_f64::<LittleEndian>(image.qx)?;
            writer.write_f64::<LittleEndian>(image.qy)?;
            writer.write_f64::<LittleEndian>(image.qz)?;
            
            // Write translation (tx, ty, tz)
            writer.write_f64::<LittleEndian>(image.tx)?;
            writer.write_f64::<LittleEndian>(image.ty)?;
            writer.write_f64::<LittleEndian>(image.tz)?;
            
            // Write camera ID
            writer.write_u32::<LittleEndian>(image.camera_id)?;
            
            // Write image name (null-terminated string)
            writer.write_all(image.name.as_bytes())?;
            writer.write_u8(0)?; // null terminator
            
            // Write 2D points
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
    
    fn write_points3d_binary(&self, output_dir: &Path, points3d: &HashMap<u64, Point3D>) -> Result<(), Box<dyn std::error::Error>> {
        let points3d_path = output_dir.join("points3D.bin");
        let file = File::create(points3d_path)?;
        let mut writer = BufWriter::new(file);
        
        // Write number of 3D points
        writer.write_u64::<LittleEndian>(points3d.len() as u64)?;
        
        // Sort points3D by ID for consistent output order (matching COLMAP reference)
        let mut sorted_points3d: Vec<_> = points3d.iter().collect();
        sorted_points3d.sort_by_key(|(point3d_id, _)| *point3d_id);
        
        for (_, point3d) in sorted_points3d {
            // Write point3D ID
            writer.write_u64::<LittleEndian>(point3d.point3d_id)?;
            
            // Write 3D coordinates
            writer.write_f64::<LittleEndian>(point3d.x)?;
            writer.write_f64::<LittleEndian>(point3d.y)?;
            writer.write_f64::<LittleEndian>(point3d.z)?;
            
            // Write color (RGB)
            writer.write_u8(point3d.r)?;
            writer.write_u8(point3d.g)?;
            writer.write_u8(point3d.b)?;
            
            // Write error
            writer.write_f64::<LittleEndian>(point3d.error)?;
            
            // Write track (list of image_id, point2D_idx pairs)
            writer.write_u64::<LittleEndian>(point3d.track.len() as u64)?;
            for track_element in &point3d.track {
                writer.write_u32::<LittleEndian>(track_element.image_id)?;
                writer.write_u32::<LittleEndian>(track_element.point2d_idx)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
}

/// Statistics from splitting operation
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

/// Python wrapper function
#[pyfunction]
pub fn split_cameras(config: CameraConfig) -> PyResult<HashMap<String, usize>> {
    let splitter = ColmapSplitter::new(config);
    
    match splitter.split_cameras() {
        Ok(stats) => {
            let mut result = HashMap::new();
            result.insert("cameras_loaded".to_string(), stats.cameras_loaded);
            result.insert("images_loaded".to_string(), stats.images_loaded);
            result.insert("points3d_loaded".to_string(), stats.points3d_loaded);
            result.insert("patches_written".to_string(), stats.patches_written);
            result.insert("total_cameras_written".to_string(), stats.total_cameras_written);
            result.insert("total_images_written".to_string(), stats.total_images_written);
            result.insert("total_points3d_written".to_string(), stats.total_points3d_written);
            Ok(result)
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Camera splitting failed: {}", e)))
    }
}