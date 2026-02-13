use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Auto,
    Rmbg14,
    Rmbg20,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OnnxVariant {
    Fp16,
    Fp32,
    Quantized,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionProvider {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuBackendPreference {
    Auto,
    DirectML,
    Cuda,
    CoreML,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PlatformTarget {
    Cli,
    Tauri,
    Android,
    Ios,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePolicy {
    pub max_inference_pixels: u32,
    pub max_latency_ms: u32,
    pub allow_rmbg20: bool,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            max_inference_pixels: 2_000_000,
            max_latency_ms: 1_500,
            allow_rmbg20: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub requested_model: ModelKind,
    pub onnx_variant: OnnxVariant,
    pub execution_provider: ExecutionProvider,
    pub gpu_backend: GpuBackendPreference,
    pub benchmark_provider: bool,
    pub emit_mask_png: bool,
    pub input_path: Option<PathBuf>,
    pub input_bytes: Option<Vec<u8>>,
    pub model_dir: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub model_used: ModelKind,
    pub mask_png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub execution_provider_selected: String,
    pub gpu_backend_selected: Option<String>,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEventType {
    LoadStart,
    LoadSuccess,
    LoadError,
    InferenceStart,
    InferenceSuccess,
    InferenceError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub event_type: TelemetryEventType,
    pub model: ModelKind,
    pub platform: PlatformTarget,
    pub duration_ms: Option<u64>,
    pub detail: Option<String>,
}

pub trait TelemetrySink: Send + Sync {
    fn emit(&self, event: TelemetryEvent);
}

pub trait InferenceBackend: Send + Sync {
    fn infer(&self, request: &InferenceRequest, selected_model: ModelKind) -> Result<InferenceResult, CoreError>;
}

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("rmbg-2.0 is disabled by runtime policy")]
    Rmbg20Disabled,
    #[error("missing input bytes and input path")]
    MissingInput,
    #[error("backend error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorCode {
    Rmbg20Disabled,
    MissingInput,
    BackendError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorInfo {
    pub code: ErrorCode,
    pub message: String,
}

impl CoreError {
    pub fn as_error_info(&self) -> ErrorInfo {
        match self {
            Self::Rmbg20Disabled => ErrorInfo {
                code: ErrorCode::Rmbg20Disabled,
                message: self.to_string(),
            },
            Self::MissingInput => ErrorInfo {
                code: ErrorCode::MissingInput,
                message: self.to_string(),
            },
            Self::Backend(message) => ErrorInfo {
                code: ErrorCode::BackendError,
                message: message.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeConfig {
    pub model: String,
    pub onnx_variant: String,
    pub execution_provider: String,
    pub gpu_backend: String,
    pub benchmark_provider: bool,
    pub model_dir: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model: "auto".to_string(),
            onnx_variant: "fp16".to_string(),
            execution_provider: "auto".to_string(),
            gpu_backend: "auto".to_string(),
            benchmark_provider: true,
            model_dir: None,
        }
    }
}

pub fn resolve_runtime_config(overrides: RuntimeConfig) -> RuntimeConfig {
    let mut cfg = RuntimeConfig::default();
    if !overrides.model.trim().is_empty() {
        cfg.model = overrides.model;
    }
    if !overrides.onnx_variant.trim().is_empty() {
        cfg.onnx_variant = overrides.onnx_variant;
    }
    if !overrides.execution_provider.trim().is_empty() {
        cfg.execution_provider = overrides.execution_provider;
    }
    if !overrides.gpu_backend.trim().is_empty() {
        cfg.gpu_backend = overrides.gpu_backend;
    }
    cfg.benchmark_provider = overrides.benchmark_provider;
    cfg.model_dir = overrides.model_dir;
    cfg
}

pub mod v1 {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct RemoveBackgroundRequest {
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
    #[serde(rename_all = "camelCase")]
    pub struct RemoveBackgroundResponse {
        pub model_used: String,
        pub width: u32,
        pub height: u32,
        pub mask_png: Vec<u8>,
        pub provider_selected: String,
        pub backend_selected: Option<String>,
        pub fallback_used: bool,
    }
}

pub fn resolve_model(request: &InferenceRequest, policy: &RuntimePolicy) -> Result<ModelKind, CoreError> {
    let pixels = request.width.saturating_mul(request.height);
    match request.requested_model {
        ModelKind::Rmbg20 if !policy.allow_rmbg20 => Err(CoreError::Rmbg20Disabled),
        ModelKind::Rmbg20 => Ok(ModelKind::Rmbg20),
        ModelKind::Rmbg14 => Ok(ModelKind::Rmbg14),
        ModelKind::Auto => {
            if policy.allow_rmbg20 && pixels <= policy.max_inference_pixels {
                Ok(ModelKind::Rmbg20)
            } else {
                Ok(ModelKind::Rmbg14)
            }
        }
    }
}

pub fn run_inference(
    backend: &dyn InferenceBackend,
    request: &InferenceRequest,
    policy: &RuntimePolicy,
) -> Result<InferenceResult, CoreError> {
    run_inference_with_telemetry(backend, request, policy, PlatformTarget::Cli, None)
}

pub fn run_inference_with_telemetry(
    backend: &dyn InferenceBackend,
    request: &InferenceRequest,
    policy: &RuntimePolicy,
    platform: PlatformTarget,
    telemetry: Option<&dyn TelemetrySink>,
) -> Result<InferenceResult, CoreError> {
    if request.input_bytes.is_none() && request.input_path.is_none() {
        return Err(CoreError::MissingInput);
    }
    let start = Instant::now();
    if let Some(sink) = telemetry {
        sink.emit(TelemetryEvent {
            event_type: TelemetryEventType::InferenceStart,
            model: request.requested_model,
            platform,
            duration_ms: None,
            detail: None,
        });
    }
    let selected_model = resolve_model(request, policy)?;
    match backend.infer(request, selected_model) {
        Ok(result) => {
            if let Some(sink) = telemetry {
                sink.emit(TelemetryEvent {
                    event_type: TelemetryEventType::InferenceSuccess,
                    model: result.model_used,
                    platform,
                    duration_ms: Some(start.elapsed().as_millis() as u64),
                    detail: Some(format!(
                        "provider={},backend={},fallback={}",
                        result.execution_provider_selected,
                        result.gpu_backend_selected.clone().unwrap_or_else(|| "none".to_string()),
                        result.fallback_used
                    )),
                });
            }
            Ok(result)
        }
        Err(err) => {
            if let Some(sink) = telemetry {
                sink.emit(TelemetryEvent {
                    event_type: TelemetryEventType::InferenceError,
                    model: selected_model,
                    platform,
                    duration_ms: Some(start.elapsed().as_millis() as u64),
                    detail: Some(err.to_string()),
                });
            }
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubBackend;

    impl InferenceBackend for StubBackend {
        fn infer(&self, request: &InferenceRequest, selected_model: ModelKind) -> Result<InferenceResult, CoreError> {
            Ok(InferenceResult {
                model_used: selected_model,
                mask_png: vec![0, 1, 2],
                width: request.width,
                height: request.height,
                execution_provider_selected: "cpu".to_string(),
                gpu_backend_selected: None,
                fallback_used: false,
            })
        }
    }

    #[test]
    fn auto_falls_back_to_rmbg14_when_pixel_budget_exceeded() {
        let request = InferenceRequest {
            requested_model: ModelKind::Auto,
            onnx_variant: OnnxVariant::Fp16,
            execution_provider: ExecutionProvider::Auto,
            gpu_backend: GpuBackendPreference::Auto,
            benchmark_provider: true,
            emit_mask_png: true,
            input_path: Some(PathBuf::from("input.png")),
            input_bytes: None,
            model_dir: None,
            width: 4096,
            height: 4096,
        };
        let policy = RuntimePolicy {
            max_inference_pixels: 1_000_000,
            max_latency_ms: 1500,
            allow_rmbg20: true,
        };
        let selected = resolve_model(&request, &policy).expect("model selection should work");
        assert_eq!(selected, ModelKind::Rmbg14);
    }

    #[test]
    fn inference_uses_selected_model() {
        let request = InferenceRequest {
            requested_model: ModelKind::Rmbg20,
            onnx_variant: OnnxVariant::Fp16,
            execution_provider: ExecutionProvider::Auto,
            gpu_backend: GpuBackendPreference::Auto,
            benchmark_provider: true,
            emit_mask_png: true,
            input_path: Some(PathBuf::from("input.png")),
            input_bytes: None,
            model_dir: None,
            width: 100,
            height: 100,
        };
        let policy = RuntimePolicy::default();
        let result = run_inference(&StubBackend, &request, &policy).expect("inference should succeed");
        assert_eq!(result.model_used, ModelKind::Rmbg20);
    }
}
