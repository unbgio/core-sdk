use anyhow::Result;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use unbg_core::ModelKind;

fn build_sample_png() -> Result<Vec<u8>> {
    let img = ImageBuffer::from_fn(8, 8, |x, y| {
        if (x + y) % 2 == 0 {
            Rgb([255, 255, 255])
        } else {
            Rgb([10, 10, 10])
        }
    });
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)?;
    Ok(out)
}

#[test]
fn adapters_produce_consistent_mask_shape() -> Result<()> {
    std::env::set_var("UNBG_ALLOW_PLACEHOLDER", "1");
    set_ort_dylib_path_if_available();
    let sample = build_sample_png()?;

    let tauri = tauri_plugin_unbg::remove_background(tauri_plugin_unbg::TauriRemoveRequest {
        image_bytes: sample.clone(),
        width: 8,
        height: 8,
        model: ModelKind::Auto,
        max_inference_pixels: 2_000_000,
        execution_provider: None,
        gpu_backend: None,
        benchmark_provider: None,
        onnx_variant: None,
        model_dir: None,
    })?;

    let android = android_unbg::process_image(android_unbg::AndroidBridgeRequest {
        image_bytes: sample.clone(),
        width: 8,
        height: 8,
        model: ModelKind::Auto,
        onnx_variant: None,
        model_dir: None,
        execution_provider: None,
        gpu_backend: None,
        benchmark_provider: None,
    })?;

    let ios = ios_unbg::process_image(ios_unbg::IosBridgeRequest {
        image_bytes: sample,
        width: 8,
        height: 8,
        model: ModelKind::Auto,
        onnx_variant: None,
        model_dir: None,
        execution_provider: None,
        gpu_backend: None,
        benchmark_provider: None,
    })?;

    assert_eq!(tauri.model_used, ModelKind::Rmbg20);
    assert_eq!(android.model_used, ModelKind::Rmbg20);
    assert_eq!(ios.model_used, ModelKind::Rmbg20);
    assert_eq!(tauri.mask_png.len(), android.mask_png.len());
    assert_eq!(android.mask_png.len(), ios.mask_png.len());
    assert!(!tauri.provider_selected.is_empty());
    assert!(!android.provider_selected.is_empty());
    assert!(!ios.provider_selected.is_empty());
    Ok(())
}

fn set_ort_dylib_path_if_available() {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }
    let lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    if let Some(python_candidate) = discover_ort_from_python(lib_name) {
        std::env::set_var("ORT_DYLIB_PATH", python_candidate);
        return;
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if dir.to_string_lossy().to_ascii_lowercase().contains("windows\\system32") {
                continue;
            }
            let candidate = dir.join(lib_name);
            if candidate.exists() {
                std::env::set_var("ORT_DYLIB_PATH", candidate);
                break;
            }
        }
    }
}

fn discover_ort_from_python(lib_name: &str) -> Option<std::path::PathBuf> {
    let probe = format!(
        "import pathlib, onnxruntime; p = pathlib.Path(onnxruntime.__file__).resolve().parent / 'capi' / '{}'; print(p if p.exists() else '')",
        lib_name
    );
    for exe in ["python", "python3", "py"] {
        let output = match std::process::Command::new(exe).arg("-c").arg(&probe).output() {
            Ok(output) => output,
            Err(_) => continue,
        };
        if !output.status.success() {
            continue;
        }
        let candidate = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if candidate.is_empty() {
            continue;
        }
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}
