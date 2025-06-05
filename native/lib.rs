use pyo3::prelude::*;

mod split;

/// The main PyO3 module for wildflow.splat
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Export split functionality
    m.add_class::<split::Config>()?;
    m.add_class::<split::Patch>()?;
    m.add_function(wrap_pyfunction!(split::split_ply, m)?)?;
    Ok(())
} 