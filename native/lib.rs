use pyo3::prelude::*;

pub mod cleanup;
mod patches;
mod split_cameras;
mod split_point_cloud;

/// The main PyO3 module for wildflow.splat
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<split_point_cloud::Config>()?;
    m.add_class::<split_point_cloud::Patch>()?;
    m.add_function(wrap_pyfunction!(split_point_cloud::split_ply, m)?)?;

    m.add_class::<patches::BoundingBox>()?;
    m.add_function(wrap_pyfunction!(patches::patches, m)?)?;

    m.add_function(wrap_pyfunction!(cleanup::cleanup_splats, m)?)?;

    m.add_function(wrap_pyfunction!(split_cameras::split_cameras_json, m)?)?;

    Ok(())
}
