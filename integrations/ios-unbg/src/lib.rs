use anyhow::Result;
use serde::{Deserialize, Serialize};
use unbg_core::{v1, ExecutionProvider, GpuBackendPreference, ModelKind};
use unbg_uniffi::{remove_background, FfiRemoveBackgroundRequest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IosBridgeRequest {
    pub image_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub model: ModelKind,
    pub onnx_variant: Option<String>,
    pub model_dir: Option<String>,
    pub execution_provider: Option<ExecutionProvider>,
    pub gpu_backend: Option<GpuBackendPreference>,
    pub benchmark_provider: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IosBridgeResponse {
    pub model_used: ModelKind,
    pub mask_png: Vec<u8>,
    pub provider_selected: String,
    pub backend_selected: Option<String>,
    pub fallback_used: bool,
}

pub fn process_image(request: IosBridgeRequest) -> Result<IosBridgeResponse> {
    let output = process_image_v1(v1::RemoveBackgroundRequest {
        image_bytes: request.image_bytes,
        width: request.width,
        height: request.height,
        model: model_label(request.model).to_string(),
        onnx_variant: request.onnx_variant,
        execution_provider: request.execution_provider.map(provider_label),
        gpu_backend: request.gpu_backend.map(gpu_backend_label),
        benchmark_provider: request.benchmark_provider,
        model_dir: request.model_dir,
        max_inference_pixels: Some(1_500_000),
    })?;
    Ok(IosBridgeResponse {
        model_used: parse_model_kind(&output.model_used)?,
        mask_png: output.mask_png,
        provider_selected: output.provider_selected,
        backend_selected: output.backend_selected,
        fallback_used: output.fallback_used,
    })
}

pub fn process_image_v1(request: v1::RemoveBackgroundRequest) -> Result<v1::RemoveBackgroundResponse> {
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
        max_inference_pixels: request.max_inference_pixels.or(Some(1_500_000)),
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

fn parse_model_kind(raw: &str) -> Result<ModelKind> {
    match raw {
        "rmbg-1.4" => Ok(ModelKind::Rmbg14),
        "rmbg-2.0" => Ok(ModelKind::Rmbg20),
        "auto" => Ok(ModelKind::Auto),
        other => Err(anyhow::anyhow!("unknown model label '{}'", other)),
    }
}

fn model_label(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Auto => "auto",
        ModelKind::Rmbg14 => "rmbg-1.4",
        ModelKind::Rmbg20 => "rmbg-2.0",
    }
}

fn provider_label(value: ExecutionProvider) -> String {
    match value {
        ExecutionProvider::Auto => "auto".to_string(),
        ExecutionProvider::Gpu => "gpu".to_string(),
        ExecutionProvider::Cpu => "cpu".to_string(),
    }
}

fn gpu_backend_label(value: GpuBackendPreference) -> String {
    match value {
        GpuBackendPreference::Auto => "auto".to_string(),
        GpuBackendPreference::DirectML => "directml".to_string(),
        GpuBackendPreference::Cuda => "cuda".to_string(),
        GpuBackendPreference::CoreML => "coreml".to_string(),
        GpuBackendPreference::Metal => "metal".to_string(),
    }
}
