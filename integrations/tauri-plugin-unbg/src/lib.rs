use anyhow::Result;
use serde::{Deserialize, Serialize};
use unbg_core::{
    run_inference_with_telemetry, v1, ExecutionProvider, GpuBackendPreference, InferenceRequest, ModelKind, OnnxVariant, PlatformTarget,
    RuntimeConfig, RuntimePolicy,
};
use unbg_image::{estimate_rgba_bytes, ImageSize};
use unbg_telemetry::sink_from_env;
use unbg_runtime_ort::LocalOrtBackend;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriRemoveRequest {
    pub image_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub model: ModelKind,
    pub max_inference_pixels: u32,
    pub execution_provider: Option<ExecutionProvider>,
    pub gpu_backend: Option<GpuBackendPreference>,
    pub benchmark_provider: Option<bool>,
    pub onnx_variant: Option<OnnxVariant>,
    pub model_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriRemoveResponse {
    pub model_used: ModelKind,
    pub mask_png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub provider_selected: String,
    pub backend_selected: Option<String>,
    pub fallback_used: bool,
}

pub fn remove_background(request: TauriRemoveRequest) -> Result<TauriRemoveResponse> {
    let runtime_cfg = unbg_core::resolve_runtime_config(RuntimeConfig {
        model: model_label(request.model).to_string(),
        onnx_variant: request
            .onnx_variant
            .map(|v| onnx_variant_label(v).to_string())
            .unwrap_or_else(|| "fp16".to_string()),
        execution_provider: request
            .execution_provider
            .map(|v| execution_provider_label(v).to_string())
            .unwrap_or_else(|| "auto".to_string()),
        gpu_backend: request
            .gpu_backend
            .map(|v| gpu_backend_label(v).to_string())
            .unwrap_or_else(|| "auto".to_string()),
        benchmark_provider: request.benchmark_provider.unwrap_or(true),
        model_dir: request.model_dir.clone(),
    });
    let backend = LocalOrtBackend::default();
    let estimated_bytes = estimate_rgba_bytes(ImageSize {
        width: request.width,
        height: request.height,
    });
    let policy = RuntimePolicy {
        max_inference_pixels: request.max_inference_pixels,
        max_latency_ms: 1_500,
        allow_rmbg20: estimated_bytes <= 64 * 1024 * 1024,
    };
    let telemetry = sink_from_env();
    let telemetry_ref = telemetry.as_ref().map(|sink| sink.as_ref());
    let inference = run_inference_with_telemetry(
        &backend,
        &InferenceRequest {
            requested_model: parse_model_alias(&runtime_cfg.model).map_err(anyhow::Error::msg)?,
            onnx_variant: parse_onnx_variant_opt(Some(&runtime_cfg.onnx_variant))
                .map_err(anyhow::Error::msg)?
                .unwrap_or(OnnxVariant::Fp16),
            execution_provider: parse_execution_provider_opt(Some(&runtime_cfg.execution_provider))
                .map_err(anyhow::Error::msg)?
                .unwrap_or(ExecutionProvider::Auto),
            gpu_backend: parse_gpu_backend_opt(Some(&runtime_cfg.gpu_backend))
                .map_err(anyhow::Error::msg)?
                .unwrap_or(GpuBackendPreference::Auto),
            benchmark_provider: runtime_cfg.benchmark_provider,
            emit_mask_png: true,
            input_path: None,
            input_bytes: Some(request.image_bytes),
            model_dir: runtime_cfg.model_dir.map(std::path::PathBuf::from),
            width: request.width,
            height: request.height,
        },
        &policy,
        PlatformTarget::Tauri,
        telemetry_ref,
    )?;
    Ok(TauriRemoveResponse {
        model_used: inference.model_used,
        mask_png: inference.mask_png,
        width: inference.width,
        height: inference.height,
        provider_selected: inference.execution_provider_selected,
        backend_selected: inference.gpu_backend_selected,
        fallback_used: inference.fallback_used,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriCommandRequest {
    pub image_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub model: Option<String>,
    pub max_inference_pixels: Option<u32>,
    pub execution_provider: Option<String>,
    pub gpu_backend: Option<String>,
    pub benchmark_provider: Option<bool>,
    pub onnx_variant: Option<String>,
    pub model_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriCommandResponse {
    pub model_used: String,
    pub width: u32,
    pub height: u32,
    pub mask_png: Vec<u8>,
    pub provider_selected: String,
    pub backend_selected: Option<String>,
    pub fallback_used: bool,
}

pub fn remove_background_command(request: TauriCommandRequest) -> std::result::Result<TauriCommandResponse, String> {
    let v1_result = remove_background_v1(v1::RemoveBackgroundRequest {
        image_bytes: request.image_bytes,
        width: request.width,
        height: request.height,
        model: request.model.unwrap_or_else(|| "auto".to_string()),
        max_inference_pixels: request.max_inference_pixels.or(Some(2_000_000)),
        execution_provider: request.execution_provider,
        gpu_backend: request.gpu_backend,
        benchmark_provider: request.benchmark_provider,
        onnx_variant: request.onnx_variant,
        model_dir: request.model_dir,
    })?;
    Ok(TauriCommandResponse {
        model_used: v1_result.model_used,
        width: v1_result.width,
        height: v1_result.height,
        mask_png: v1_result.mask_png,
        provider_selected: v1_result.provider_selected,
        backend_selected: v1_result.backend_selected,
        fallback_used: v1_result.fallback_used,
    })
}

pub fn remove_background_v1(request: v1::RemoveBackgroundRequest) -> std::result::Result<v1::RemoveBackgroundResponse, String> {
    let response = remove_background(TauriRemoveRequest {
        image_bytes: request.image_bytes,
        width: request.width,
        height: request.height,
        model: parse_model_alias(&request.model)?,
        max_inference_pixels: request.max_inference_pixels.unwrap_or(2_000_000),
        execution_provider: parse_execution_provider_opt(request.execution_provider.as_deref())?,
        gpu_backend: parse_gpu_backend_opt(request.gpu_backend.as_deref())?,
        benchmark_provider: request.benchmark_provider,
        onnx_variant: parse_onnx_variant_opt(request.onnx_variant.as_deref())?,
        model_dir: request.model_dir,
    })
    .map_err(|err| err.to_string())?;
    Ok(v1::RemoveBackgroundResponse {
        model_used: model_label(response.model_used).to_string(),
        width: response.width,
        height: response.height,
        mask_png: response.mask_png,
        provider_selected: response.provider_selected,
        backend_selected: response.backend_selected,
        fallback_used: response.fallback_used,
    })
}

#[cfg(feature = "tauri-plugin")]
#[tauri::command]
fn tauri_remove_background_command(request: TauriCommandRequest) -> std::result::Result<TauriCommandResponse, String> {
    remove_background_command(request)
}

#[cfg(feature = "tauri-plugin")]
pub fn init<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("unbg")
        .invoke_handler(tauri::generate_handler![tauri_remove_background_command])
        .build()
}

fn parse_model_alias(raw: &str) -> std::result::Result<ModelKind, String> {
    match raw.to_ascii_lowercase().as_str() {
        "auto" => Ok(ModelKind::Auto),
        "fast" | "rmbg-1.4" => Ok(ModelKind::Rmbg14),
        "quality" | "rmbg-2.0" => Ok(ModelKind::Rmbg20),
        other => Err(format!(
            "unknown model '{}'; expected one of: auto, fast, quality, rmbg-1.4, rmbg-2.0",
            other
        )),
    }
}

fn parse_execution_provider_opt(raw: Option<&str>) -> std::result::Result<Option<ExecutionProvider>, String> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(ExecutionProvider::Auto)),
            "gpu" => Ok(Some(ExecutionProvider::Gpu)),
            "cpu" => Ok(Some(ExecutionProvider::Cpu)),
            other => Err(format!(
                "unknown execution provider '{}'; expected one of: auto, gpu, cpu",
                other
            )),
        },
    }
}

fn parse_gpu_backend_opt(raw: Option<&str>) -> std::result::Result<Option<GpuBackendPreference>, String> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(GpuBackendPreference::Auto)),
            "directml" => Ok(Some(GpuBackendPreference::DirectML)),
            "cuda" => Ok(Some(GpuBackendPreference::Cuda)),
            "coreml" => Ok(Some(GpuBackendPreference::CoreML)),
            "metal" => Ok(Some(GpuBackendPreference::Metal)),
            other => Err(format!(
                "unknown gpu backend '{}'; expected one of: auto, directml, cuda, coreml, metal",
                other
            )),
        },
    }
}

fn parse_onnx_variant_opt(raw: Option<&str>) -> std::result::Result<Option<OnnxVariant>, String> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(OnnxVariant::Auto)),
            "fp16" => Ok(Some(OnnxVariant::Fp16)),
            "fp32" => Ok(Some(OnnxVariant::Fp32)),
            "quantized" | "q8" => Ok(Some(OnnxVariant::Quantized)),
            other => Err(format!(
                "unknown onnx variant '{}'; expected one of: auto, fp16, fp32, quantized",
                other
            )),
        },
    }
}

fn model_label(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Auto => "auto",
        ModelKind::Rmbg14 => "rmbg-1.4",
        ModelKind::Rmbg20 => "rmbg-2.0",
    }
}

fn onnx_variant_label(value: OnnxVariant) -> &'static str {
    match value {
        OnnxVariant::Auto => "auto",
        OnnxVariant::Fp16 => "fp16",
        OnnxVariant::Fp32 => "fp32",
        OnnxVariant::Quantized => "quantized",
    }
}

fn execution_provider_label(value: ExecutionProvider) -> &'static str {
    match value {
        ExecutionProvider::Auto => "auto",
        ExecutionProvider::Gpu => "gpu",
        ExecutionProvider::Cpu => "cpu",
    }
}

fn gpu_backend_label(value: GpuBackendPreference) -> &'static str {
    match value {
        GpuBackendPreference::Auto => "auto",
        GpuBackendPreference::DirectML => "directml",
        GpuBackendPreference::Cuda => "cuda",
        GpuBackendPreference::CoreML => "coreml",
        GpuBackendPreference::Metal => "metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

    fn sample_png() -> Vec<u8> {
        let img = ImageBuffer::from_fn(8, 8, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([255, 255, 255])
            } else {
                Rgb([10, 10, 10])
            }
        });
        let mut out = Vec::new();
        DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)
            .expect("sample png");
        out
    }

    fn set_ort_dylib_path_if_available() {
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            return;
        }
        let candidate = "C:/Users/jefff/AppData/Roaming/Python/Python312/site-packages/onnxruntime/capi/onnxruntime.dll";
        if std::path::Path::new(candidate).exists() {
            std::env::set_var("ORT_DYLIB_PATH", candidate);
        }
    }

    #[test]
    fn command_maps_fast_alias() {
        std::env::set_var("UNBG_ALLOW_PLACEHOLDER", "1");
        set_ort_dylib_path_if_available();
        let response = remove_background_command(TauriCommandRequest {
            image_bytes: sample_png(),
            width: 8,
            height: 8,
            model: Some("fast".to_string()),
            max_inference_pixels: None,
            execution_provider: None,
            gpu_backend: None,
            benchmark_provider: None,
            onnx_variant: Some("fp16".to_string()),
            model_dir: None,
        })
        .expect("command should succeed");

        assert_eq!(response.model_used, "rmbg-1.4");
        assert_eq!(response.width, 8);
        assert_eq!(response.height, 8);
        assert!(!response.mask_png.is_empty());
        assert!(!response.provider_selected.is_empty());
    }

    #[test]
    fn command_rejects_invalid_model() {
        let error = remove_background_command(TauriCommandRequest {
            image_bytes: vec![1, 2, 3],
            width: 1,
            height: 1,
            model: Some("bad-model".to_string()),
            max_inference_pixels: None,
            execution_provider: None,
            gpu_backend: None,
            benchmark_provider: None,
            onnx_variant: None,
            model_dir: None,
        })
        .expect_err("should fail for invalid model");

        assert!(error.contains("unknown model"));
    }
}
