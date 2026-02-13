use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use unbg_core::{
    run_inference_with_telemetry, v1, CoreError, ErrorInfo, ExecutionProvider, GpuBackendPreference, InferenceRequest, ModelKind,
    OnnxVariant, PlatformTarget, RuntimeConfig, RuntimePolicy,
};
use unbg_image::{estimate_rgba_bytes, ImageSize};
use unbg_model_registry::default_model_dir;
use unbg_telemetry::sink_from_env;
use unbg_runtime_ort::LocalOrtBackend;

uniffi::setup_scaffolding!();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiRemoveBackgroundRequest {
    pub image_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub model: String,
    pub onnx_variant: Option<String>,
    pub execution_provider: Option<String>,
    pub gpu_backend: Option<String>,
    pub benchmark_provider: Option<bool>,
    pub model_dir: Option<String>,
    pub max_inference_pixels: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiRemoveBackgroundResponse {
    pub model_used: String,
    pub width: u32,
    pub height: u32,
    pub mask_png: Vec<u8>,
    pub provider_selected: String,
    pub backend_selected: Option<String>,
    pub fallback_used: bool,
}

#[derive(Debug, Error, uniffi::Error)]
pub enum FfiError {
    #[error("invalid-argument")]
    InvalidArgument,
    #[error("inference")]
    Inference,
}

#[derive(uniffi::Object)]
pub struct UnbgApi;

#[uniffi::export]
impl UnbgApi {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self
    }

    pub fn remove_background_v1_json(&self, request_json: String) -> String {
        let request: v1::RemoveBackgroundRequest = match serde_json::from_str(&request_json) {
            Ok(request) => request,
            Err(_) => return "{\"code\":\"invalid-argument\",\"message\":\"invalid request json\"}".to_string(),
        };
        match remove_background_v1(request) {
            Ok(response) => serde_json::to_string(&response)
                .unwrap_or_else(|_| "{\"code\":\"inference\",\"message\":\"response encode failed\"}".to_string()),
            Err(err) => format!("{{\"code\":\"{}\",\"message\":\"{}\"}}", error_code(&err), err),
        }
    }

    pub fn default_model_dir_string(&self) -> String {
        match default_model_dir_string() {
            Ok(path) => path,
            Err(err) => format!("{{\"code\":\"{}\",\"message\":\"{}\"}}", error_code(&err), err),
        }
    }

    pub fn supported_model_aliases_json(&self) -> String {
        serde_json::to_string(&supported_model_aliases()).unwrap_or_else(|_| "[]".to_string())
    }
}

pub fn remove_background(request: FfiRemoveBackgroundRequest) -> Result<FfiRemoveBackgroundResponse, FfiError> {
    let runtime_cfg = unbg_core::resolve_runtime_config(RuntimeConfig {
        model: request.model.clone(),
        onnx_variant: request.onnx_variant.clone().unwrap_or_else(|| "fp16".to_string()),
        execution_provider: request.execution_provider.clone().unwrap_or_else(|| "auto".to_string()),
        gpu_backend: request.gpu_backend.clone().unwrap_or_else(|| "auto".to_string()),
        benchmark_provider: request.benchmark_provider.unwrap_or(true),
        model_dir: request.model_dir.clone(),
    });
    let backend = LocalOrtBackend::default();
    let estimated_bytes = estimate_rgba_bytes(ImageSize {
        width: request.width,
        height: request.height,
    });
    let telemetry = sink_from_env();
    let telemetry_ref = telemetry.as_ref().map(|sink| sink.as_ref());
    let inference = run_inference_with_telemetry(
        &backend,
        &InferenceRequest {
            requested_model: parse_model_alias(&runtime_cfg.model)?,
            onnx_variant: parse_onnx_variant_opt(Some(&runtime_cfg.onnx_variant))?.unwrap_or(OnnxVariant::Fp16),
            execution_provider: parse_execution_provider_opt(Some(&runtime_cfg.execution_provider))?
                .unwrap_or(ExecutionProvider::Auto),
            gpu_backend: parse_gpu_backend_opt(Some(&runtime_cfg.gpu_backend))?.unwrap_or(GpuBackendPreference::Auto),
            benchmark_provider: runtime_cfg.benchmark_provider,
            emit_mask_png: true,
            input_path: None,
            input_bytes: Some(request.image_bytes),
            model_dir: runtime_cfg.model_dir.map(PathBuf::from),
            width: request.width,
            height: request.height,
        },
        &RuntimePolicy {
            max_inference_pixels: request.max_inference_pixels.unwrap_or(2_000_000),
            max_latency_ms: 1_500,
            allow_rmbg20: estimated_bytes <= 64 * 1024 * 1024,
        },
        PlatformTarget::Cli,
        telemetry_ref,
    )
    .map_err(map_core_error)?;

    Ok(FfiRemoveBackgroundResponse {
        model_used: model_label(inference.model_used).to_string(),
        width: inference.width,
        height: inference.height,
        mask_png: inference.mask_png,
        provider_selected: inference.execution_provider_selected,
        backend_selected: inference.gpu_backend_selected,
        fallback_used: inference.fallback_used,
    })
}

pub fn supported_model_aliases() -> Vec<String> {
    vec![
        "auto".to_string(),
        "fast".to_string(),
        "quality".to_string(),
        "rmbg-1.4".to_string(),
        "rmbg-2.0".to_string(),
    ]
}

pub fn remove_background_v1(request: v1::RemoveBackgroundRequest) -> Result<v1::RemoveBackgroundResponse, FfiError> {
    let out = remove_background(FfiRemoveBackgroundRequest {
        image_bytes: request.image_bytes,
        width: request.width,
        height: request.height,
        model: request.model,
        onnx_variant: request.onnx_variant,
        execution_provider: request.execution_provider,
        gpu_backend: request.gpu_backend,
        benchmark_provider: request.benchmark_provider,
        model_dir: request.model_dir,
        max_inference_pixels: request.max_inference_pixels,
    })?;
    Ok(v1::RemoveBackgroundResponse {
        model_used: out.model_used,
        width: out.width,
        height: out.height,
        mask_png: out.mask_png,
        provider_selected: out.provider_selected,
        backend_selected: out.backend_selected,
        fallback_used: out.fallback_used,
    })
}

pub fn default_model_dir_string() -> Result<String, FfiError> {
    let path = default_model_dir().map_err(|_err| FfiError::Inference)?;
    Ok(path.display().to_string())
}

fn parse_model_alias(raw: &str) -> Result<ModelKind, FfiError> {
    match raw.to_ascii_lowercase().as_str() {
        "auto" => Ok(ModelKind::Auto),
        "fast" | "rmbg-1.4" => Ok(ModelKind::Rmbg14),
        "quality" | "rmbg-2.0" => Ok(ModelKind::Rmbg20),
        _other => Err(FfiError::InvalidArgument),
    }
}

fn parse_onnx_variant_opt(raw: Option<&str>) -> Result<Option<OnnxVariant>, FfiError> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(OnnxVariant::Auto)),
            "fp16" => Ok(Some(OnnxVariant::Fp16)),
            "fp32" => Ok(Some(OnnxVariant::Fp32)),
            "quantized" | "q8" => Ok(Some(OnnxVariant::Quantized)),
            _other => Err(FfiError::InvalidArgument),
        },
    }
}

fn parse_execution_provider_opt(raw: Option<&str>) -> Result<Option<ExecutionProvider>, FfiError> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(ExecutionProvider::Auto)),
            "gpu" => Ok(Some(ExecutionProvider::Gpu)),
            "cpu" => Ok(Some(ExecutionProvider::Cpu)),
            _other => Err(FfiError::InvalidArgument),
        },
    }
}

fn parse_gpu_backend_opt(raw: Option<&str>) -> Result<Option<GpuBackendPreference>, FfiError> {
    match raw.map(|value| value.to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            "auto" => Ok(Some(GpuBackendPreference::Auto)),
            "directml" => Ok(Some(GpuBackendPreference::DirectML)),
            "cuda" => Ok(Some(GpuBackendPreference::Cuda)),
            "coreml" => Ok(Some(GpuBackendPreference::CoreML)),
            "metal" => Ok(Some(GpuBackendPreference::Metal)),
            _other => Err(FfiError::InvalidArgument),
        },
    }
}

fn map_core_error(err: CoreError) -> FfiError {
    let info: ErrorInfo = err.as_error_info();
    match info.code {
        unbg_core::ErrorCode::MissingInput => FfiError::InvalidArgument,
        _ => FfiError::Inference,
    }
}

fn model_label(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Auto => "auto",
        ModelKind::Rmbg14 => "rmbg-1.4",
        ModelKind::Rmbg20 => "rmbg-2.0",
    }
}

fn error_code(err: &FfiError) -> &'static str {
    match err {
        FfiError::InvalidArgument => "invalid-argument",
        FfiError::Inference => "inference",
    }
}

