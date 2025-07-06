use pyo3::prelude::*;

mod split;
mod patches;
pub mod cleanup;

/// The main PyO3 module for wildflow.splat
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Export split functionality
    m.add_class::<split::Config>()?;
    m.add_class::<split::Patch>()?;
    m.add_function(wrap_pyfunction!(split::split_ply, m)?)?;
    
    // Export patches functionality
    m.add_class::<patches::BoundingBox>()?;
    m.add_function(wrap_pyfunction!(patches::patches, m)?)?;

    // Export cleanup functionality
    m.add_class::<cleanup::CleanConfig>()?;
    m.add_function(wrap_pyfunction!(cleanup::cleanup_ply, m)?)?;
    
    Ok(())
}
 