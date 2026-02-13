use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage, ImageFormat, Luma};
use ort::{inputs, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use unbg_core::{
    CoreError, ExecutionProvider, GpuBackendPreference, InferenceBackend, InferenceRequest, InferenceResult, ModelKind, OnnxVariant,
};
use unbg_model_registry::{model_revision_dir, read_lockfile, resolve_model_paths, KnownModel};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct RuntimeDescriptor {
    pub execution_provider: String,
}

#[derive(Debug, Clone)]
pub struct LocalOrtBackend {
    descriptor: RuntimeDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ProviderChoice {
    Cpu,
    DirectML,
    Cuda,
    CoreML,
}

static AUTO_PROVIDER_CACHE: OnceLock<Mutex<std::collections::HashMap<String, ProviderChoice>>> = OnceLock::new();
thread_local! {
    static SESSION_CACHE: RefCell<std::collections::HashMap<String, Session>> = RefCell::new(std::collections::HashMap::new());
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct PersistedProviderCache {
    providers: std::collections::HashMap<String, String>,
}

impl Default for LocalOrtBackend {
    fn default() -> Self {
        Self {
            descriptor: RuntimeDescriptor {
                execution_provider: "cpu".to_string(),
            },
        }
    }
}

impl LocalOrtBackend {
    pub fn descriptor(&self) -> &RuntimeDescriptor {
        &self.descriptor
    }

    fn load_image(&self, request: &InferenceRequest) -> Result<DynamicImage, CoreError> {
        if let Some(bytes) = &request.input_bytes {
            return image::load_from_memory(bytes).map_err(|e| CoreError::Backend(e.to_string()));
        }
        if let Some(path) = &request.input_path {
            let bytes = fs::read(path).map_err(|e| CoreError::Backend(e.to_string()))?;
            return image::load_from_memory(&bytes).map_err(|e| CoreError::Backend(e.to_string()));
        }
        Err(CoreError::MissingInput)
    }

    fn infer_fallback(
        &self,
        selected_model: ModelKind,
        image: DynamicImage,
    ) -> Result<InferenceResult, CoreError> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut mask = GrayImage::new(width, height);
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let brightness = ((pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3) as u8;
            let alpha = if brightness > 25 { 255 } else { 0 };
            mask.put_pixel(x, y, Luma([alpha]));
        }
        let mut encoded = Vec::new();
        DynamicImage::ImageLuma8(mask)
            .write_to(&mut std::io::Cursor::new(&mut encoded), ImageFormat::Png)
            .map_err(|e| CoreError::Backend(e.to_string()))?;
        Ok(InferenceResult {
            model_used: selected_model,
            mask_png: encoded,
            width,
            height,
            execution_provider_selected: "cpu".to_string(),
            gpu_backend_selected: None,
            fallback_used: false,
        })
    }
}

impl InferenceBackend for LocalOrtBackend {
    fn infer(&self, request: &InferenceRequest, selected_model: ModelKind) -> Result<InferenceResult, CoreError> {
        let image = match self.load_image(request) {
            Ok(img) => img,
            Err(err) => {
                if placeholder_fallback_allowed() {
                    return self.infer_fallback(selected_model, DynamicImage::new_rgb8(request.width.max(1), request.height.max(1)));
                }
                return Err(err);
            }
        };
        let model_file = match resolve_model_onnx_file(request, selected_model) {
            Ok(path) => path,
            Err(err) => {
                if placeholder_fallback_allowed() {
                    return self.infer_fallback(selected_model, image);
                }
                return Err(err);
            }
        };
        let candidates = candidate_providers(request);
        if candidates.is_empty() {
            return Err(CoreError::Backend("no execution providers available".to_string()));
        }

        let result = if request.execution_provider == ExecutionProvider::Auto {
            if request.benchmark_provider {
                run_auto_bench_path(&image, &model_file, selected_model, request, &candidates)
            } else {
                run_auto_cached_path(&image, &model_file, selected_model, request, &candidates)
            }
        } else {
            run_sequential_path(&image, &model_file, selected_model, &candidates, request.emit_mask_png)
        };

        match result {
            Ok(res) => Ok(res),
            Err(err) => {
                if placeholder_fallback_allowed() {
                    self.infer_fallback(selected_model, image)
                } else {
                    Err(err)
                }
            }
        }
    }
}

fn placeholder_fallback_allowed() -> bool {
    match env::var("UNBG_ALLOW_PLACEHOLDER") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        }
        Err(_) => false,
    }
}

fn resolve_model_onnx_file(request: &InferenceRequest, selected_model: ModelKind) -> Result<PathBuf, CoreError> {
    let paths = resolve_model_paths(request.model_dir.as_deref()).map_err(|e| CoreError::Backend(e.to_string()))?;
    let lock = read_lockfile(&paths).map_err(|e| CoreError::Backend(e.to_string()))?;
    let wanted_id = match selected_model {
        ModelKind::Rmbg14 => KnownModel::Rmbg14.model_id(),
        ModelKind::Rmbg20 => KnownModel::Rmbg20.model_id(),
        ModelKind::Auto => return Err(CoreError::Backend("auto model cannot resolve onnx directly".to_string())),
    };
    let model = lock
        .models
        .iter()
        .find(|m| m.model_id == wanted_id)
        .ok_or_else(|| CoreError::Backend(format!("model not found in lockfile: {}", wanted_id)))?;
    let known_model = KnownModel::from_model_id(&model.model_id)
        .ok_or_else(|| CoreError::Backend(format!("unknown model id: {}", model.model_id)))?;
    let rev_dir = model_revision_dir(&paths, known_model, &model.revision);
    find_preferred_onnx_file(&rev_dir, request.onnx_variant).ok_or_else(|| {
        CoreError::Backend(format!(
            "no .onnx file found for {} revision {} in {}",
            model.model_id,
            model.revision,
            rev_dir.display()
        ))
    })
}

fn run_sequential_path(
    image: &DynamicImage,
    model_file: &Path,
    selected_model: ModelKind,
    candidates: &[ProviderChoice],
    emit_mask_png: bool,
) -> Result<InferenceResult, CoreError> {
    let preferred = candidates[0];
    let mut errors = Vec::new();
    for provider in candidates {
        match run_provider(image, model_file, selected_model, *provider, emit_mask_png) {
            Ok((mut result, _)) => {
                result.fallback_used = *provider != preferred;
                return Ok(result);
            }
            Err(err) => errors.push(format!("{}: {}", provider_label(*provider), err)),
        }
    }
    Err(backend_error(
        "provider-exhausted",
        format!("all providers failed: {}", errors.join(" | ")),
    ))
}

fn run_auto_bench_path(
    image: &DynamicImage,
    model_file: &Path,
    selected_model: ModelKind,
    request: &InferenceRequest,
    candidates: &[ProviderChoice],
) -> Result<InferenceResult, CoreError> {
    let cache_key = provider_cache_key(selected_model, request);
    let cache = AUTO_PROVIDER_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(cached) = load_cached_provider(&cache_key, request.model_dir.as_deref()) {
        if let Ok((result, _)) = run_provider(image, model_file, selected_model, cached, request.emit_mask_png) {
            return Ok(result);
        }
    }

    let mut best: Option<(InferenceResult, ProviderChoice, u128)> = None;
    let mut errors = Vec::new();
    for provider in candidates {
        match run_provider(image, model_file, selected_model, *provider, request.emit_mask_png) {
            Ok((result, elapsed_ms)) => {
                if let Some((_, _, best_ms)) = &best {
                    if elapsed_ms < *best_ms {
                        best = Some((result, *provider, elapsed_ms));
                    }
                } else {
                    best = Some((result, *provider, elapsed_ms));
                }
            }
            Err(err) => errors.push(format!("{}: {}", provider_label(*provider), err)),
        }
    }

    if let Some((result, provider, _)) = best {
        cache
            .lock()
            .expect("provider cache lock poisoned")
            .insert(cache_key.clone(), provider);
        persist_cached_provider(&cache_key, provider, request.model_dir.as_deref());
        return Ok(result);
    }

    Err(backend_error(
        "benchmark-failed",
        format!("auto provider benchmark failed: {}", errors.join(" | ")),
    ))
}

fn run_auto_cached_path(
    image: &DynamicImage,
    model_file: &Path,
    selected_model: ModelKind,
    request: &InferenceRequest,
    candidates: &[ProviderChoice],
) -> Result<InferenceResult, CoreError> {
    let cache_key = provider_cache_key(selected_model, request);
    if let Some(cached) = load_cached_provider(&cache_key, request.model_dir.as_deref()) {
        if candidates.contains(&cached) {
            if let Ok((result, _)) = run_provider(image, model_file, selected_model, cached, request.emit_mask_png) {
                return Ok(result);
            }
        }
    }

    let mut errors = Vec::new();
    for provider in candidates {
        match run_provider(image, model_file, selected_model, *provider, request.emit_mask_png) {
            Ok((result, _)) => {
                persist_cached_provider(&cache_key, *provider, request.model_dir.as_deref());
                return Ok(result);
            }
            Err(err) => errors.push(format!("{}: {}", provider_label(*provider), err)),
        }
    }

    Err(backend_error(
        "auto-provider-failed",
        format!("all providers failed: {}", errors.join(" | ")),
    ))
}

fn run_provider(
    image: &DynamicImage,
    model_file: &Path,
    selected_model: ModelKind,
    provider: ProviderChoice,
    emit_mask_png: bool,
) -> Result<(InferenceResult, u128)> {
    let session_key = session_cache_key(model_file, provider);
    let start = Instant::now();
    let mask_png = SESSION_CACHE.with(|cache| {
        let mut cache_ref = cache.borrow_mut();
        if !cache_ref.contains_key(&session_key) {
            let session = build_session_for_provider(model_file, provider)?;
            cache_ref.insert(session_key.clone(), session);
        }
        let session = cache_ref
            .get_mut(&session_key)
            .ok_or_else(|| anyhow!("session cache failed to initialize"))?;
        run_onnx_inference(image, session, emit_mask_png)
    })
    .map_err(|e| anyhow!(e.to_string()))?;
    let elapsed = start.elapsed().as_millis();
    let (execution_provider_selected, gpu_backend_selected) = match provider {
        ProviderChoice::Cpu => ("cpu".to_string(), None),
        ProviderChoice::DirectML => ("gpu".to_string(), Some("directml".to_string())),
        ProviderChoice::Cuda => ("gpu".to_string(), Some("cuda".to_string())),
        ProviderChoice::CoreML => ("gpu".to_string(), Some("coreml".to_string())),
    };
    Ok((
        InferenceResult {
            model_used: selected_model,
            mask_png,
            width: image.width(),
            height: image.height(),
            execution_provider_selected,
            gpu_backend_selected,
            fallback_used: false,
        },
        elapsed,
    ))
}

fn session_cache_key(model_file: &Path, provider: ProviderChoice) -> String {
    format!(
        "{}|{}|{}",
        model_file.display(),
        provider_label(provider),
        std::env::var("ORT_DYLIB_PATH").unwrap_or_default()
    )
}

fn load_cached_provider(cache_key: &str, model_dir: Option<&Path>) -> Option<ProviderChoice> {
    let memory_cache = AUTO_PROVIDER_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(provider) = memory_cache
        .lock()
        .expect("provider cache lock poisoned")
        .get(cache_key)
        .copied()
    {
        return Some(provider);
    }

    let cache_path = provider_cache_file(model_dir)?;
    let raw = fs::read_to_string(cache_path).ok()?;
    let parsed: PersistedProviderCache = serde_json::from_str(&raw).ok()?;
    let provider = parsed.providers.get(cache_key).and_then(|v| parse_provider_choice(v))?;
    memory_cache
        .lock()
        .expect("provider cache lock poisoned")
        .insert(cache_key.to_string(), provider);
    Some(provider)
}

fn persist_cached_provider(cache_key: &str, provider: ProviderChoice, model_dir: Option<&Path>) {
    let memory_cache = AUTO_PROVIDER_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    memory_cache
        .lock()
        .expect("provider cache lock poisoned")
        .insert(cache_key.to_string(), provider);

    let Some(cache_path) = provider_cache_file(model_dir) else {
        return;
    };
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let existing = fs::read_to_string(&cache_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<PersistedProviderCache>(&raw).ok())
        .unwrap_or_default();
    let mut updated = existing;
    updated
        .providers
        .insert(cache_key.to_string(), provider_label(provider).to_string());
    if let Ok(serialized) = serde_json::to_string_pretty(&updated) {
        let _ = fs::write(cache_path, serialized);
    }
}

fn provider_cache_file(model_dir: Option<&Path>) -> Option<PathBuf> {
    resolve_model_paths(model_dir)
        .ok()
        .map(|paths| paths.root.join("cache").join("provider-selection.json"))
}

fn parse_provider_choice(value: &str) -> Option<ProviderChoice> {
    match value {
        "cpu" => Some(ProviderChoice::Cpu),
        "directml" => Some(ProviderChoice::DirectML),
        "cuda" => Some(ProviderChoice::Cuda),
        "coreml" => Some(ProviderChoice::CoreML),
        _ => None,
    }
}

fn provider_cache_key(selected_model: ModelKind, request: &InferenceRequest) -> String {
    let model = match selected_model {
        ModelKind::Rmbg14 => "rmbg14",
        ModelKind::Rmbg20 => "rmbg20",
        ModelKind::Auto => "auto",
    };
    let variant = match request.onnx_variant {
        OnnxVariant::Fp16 => "fp16",
        OnnxVariant::Fp32 => "fp32",
        OnnxVariant::Quantized => "quantized",
        OnnxVariant::Auto => "auto",
    };
    let fingerprint = format!(
        "{}|{}|{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::var("ORT_DYLIB_PATH").unwrap_or_default()
    );
    format!("{}|{}|{}", model, variant, fingerprint)
}

fn candidate_providers(request: &InferenceRequest) -> Vec<ProviderChoice> {
    let mut out = Vec::new();
    match request.execution_provider {
        ExecutionProvider::Cpu => out.push(ProviderChoice::Cpu),
        ExecutionProvider::Gpu => {
            out.extend(gpu_candidates(request.gpu_backend));
            out.push(ProviderChoice::Cpu);
        }
        ExecutionProvider::Auto => {
            out.extend(gpu_candidates(request.gpu_backend));
            out.push(ProviderChoice::Cpu);
        }
    }
    dedup_providers(out)
}

fn gpu_candidates(pref: GpuBackendPreference) -> Vec<ProviderChoice> {
    let mut providers = Vec::new();
    match pref {
        GpuBackendPreference::DirectML => providers.push(ProviderChoice::DirectML),
        GpuBackendPreference::Cuda => providers.push(ProviderChoice::Cuda),
        GpuBackendPreference::CoreML | GpuBackendPreference::Metal => providers.push(ProviderChoice::CoreML),
        GpuBackendPreference::Auto => {
            #[cfg(target_os = "windows")]
            {
                if cuda_likely_available() {
                    providers.push(ProviderChoice::Cuda);
                }
                providers.push(ProviderChoice::DirectML);
            }
            #[cfg(target_os = "linux")]
            {
                if cuda_likely_available() {
                    providers.push(ProviderChoice::Cuda);
                }
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                providers.push(ProviderChoice::CoreML);
            }
        }
    }
    providers
}

fn cuda_likely_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        let system32 = std::env::var("WINDIR")
            .ok()
            .map(|d| PathBuf::from(d).join("System32").join("nvcuda.dll"));
        if let Some(candidate) = system32 {
            if candidate.exists() {
                return true;
            }
        }
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                if dir.join("nvcuda.dll").exists() {
                    return true;
                }
            }
        }
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
        ];
        return candidates.iter().any(|p| Path::new(p).exists());
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

fn dedup_providers(list: Vec<ProviderChoice>) -> Vec<ProviderChoice> {
    let mut out = Vec::new();
    for provider in list {
        if !out.contains(&provider) {
            out.push(provider);
        }
    }
    out
}

fn provider_label(provider: ProviderChoice) -> &'static str {
    match provider {
        ProviderChoice::Cpu => "cpu",
        ProviderChoice::DirectML => "directml",
        ProviderChoice::Cuda => "cuda",
        ProviderChoice::CoreML => "coreml",
    }
}

fn build_session_for_provider(model_file: &Path, provider: ProviderChoice) -> Result<Session> {
    match provider {
        ProviderChoice::Cpu => Session::builder()?.commit_from_file(model_file).map_err(Into::into),
        ProviderChoice::DirectML => {
            #[cfg(feature = "directml")]
            {
                Session::builder()?
                    .with_execution_providers([ort::ep::DirectML::default().build()])?
                    .commit_from_file(model_file)
                    .map_err(Into::into)
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(anyhow!("directml feature not enabled"))
            }
        }
        ProviderChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Session::builder()?
                    .with_execution_providers([ort::ep::CUDA::default().build()])?
                    .commit_from_file(model_file)
                    .map_err(Into::into)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!("cuda feature not enabled"))
            }
        }
        ProviderChoice::CoreML => {
            #[cfg(feature = "coreml")]
            {
                Session::builder()?
                    .with_execution_providers([ort::ep::CoreML::default().build()])?
                    .commit_from_file(model_file)
                    .map_err(Into::into)
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err(anyhow!("coreml feature not enabled"))
            }
        }
    }
}

fn backend_error(kind: &str, message: String) -> CoreError {
    CoreError::Backend(format!("{}: {}", kind, message))
}

fn find_preferred_onnx_file(base_dir: &Path, onnx_variant: OnnxVariant) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = WalkDir::new(base_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| p.extension().map(|e| e == "onnx").unwrap_or(false))
        .collect();
    candidates.sort_by_key(|p| {
        let lower = p.to_string_lossy().to_lowercase();
        match onnx_variant {
            OnnxVariant::Fp16 => {
                if lower.contains("model_fp16.onnx") {
                    0
                } else if lower.contains("model.onnx") {
                    1
                } else if lower.contains("quantized") || lower.contains("q8") {
                    2
                } else {
                    3
                }
            }
            OnnxVariant::Fp32 => {
                if lower.contains("model.onnx") && !lower.contains("fp16") && !lower.contains("quantized") {
                    0
                } else if lower.contains("model_fp16.onnx") {
                    1
                } else if lower.contains("quantized") || lower.contains("q8") {
                    2
                } else {
                    3
                }
            }
            OnnxVariant::Quantized => {
                if lower.contains("quantized") || lower.contains("q8") {
                    0
                } else if lower.contains("model_fp16.onnx") {
                    1
                } else if lower.contains("model.onnx") {
                    2
                } else {
                    3
                }
            }
            OnnxVariant::Auto => {
                if lower.contains("model_fp16.onnx") {
                    0
                } else if lower.contains("model.onnx") {
                    1
                } else if lower.contains("quantized") || lower.contains("q8") {
                    2
                } else {
                    3
                }
            }
        }
    });
    candidates.into_iter().next()
}

fn run_onnx_inference(image: &DynamicImage, session: &mut Session, emit_mask_png: bool) -> Result<Vec<u8>> {
    let orig_w = image.width();
    let orig_h = image.height();
    let input_size = 1024u32;
    let resized = image.resize_exact(input_size, input_size, FilterType::Triangle).to_rgb8();

    let mut input_data = vec![0f32; (1 * 3 * input_size as usize * input_size as usize) as usize];
    for y in 0..input_size as usize {
        for x in 0..input_size as usize {
            let p = resized.get_pixel(x as u32, y as u32);
            let idx = y * input_size as usize + x;
            // RMBG-1.4 preprocessing aligns with BRIA utilities:
            // image = (pixel/255.0) - 0.5 for each channel.
            input_data[idx] = (p[0] as f32 / 255.0) - 0.5;
            input_data[input_size as usize * input_size as usize + idx] = (p[1] as f32 / 255.0) - 0.5;
            input_data[2 * input_size as usize * input_size as usize + idx] = (p[2] as f32 / 255.0) - 0.5;
        }
    }

    let input_tensor = Tensor::<f32>::from_array((
        [1usize, 3, input_size as usize, input_size as usize],
        input_data,
    ))?;
    let outputs = session.run(inputs![input_tensor])?;
    if outputs.len() == 0 {
        return Err(anyhow!("model returned no outputs"));
    }
    if !emit_mask_png {
        return Ok(Vec::new());
    }
    let view = outputs[0].try_extract_array::<f32>()?;

    let (mask_h, mask_w) = match view.ndim() {
        4 => (view.shape()[2], view.shape()[3]),
        3 => (view.shape()[1], view.shape()[2]),
        2 => (view.shape()[0], view.shape()[1]),
        _ => return Err(anyhow!("unsupported output dimensions: {:?}", view.shape())),
    };

    let mut raw = Vec::with_capacity(mask_w * mask_h);
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for y in 0..mask_h {
        for x in 0..mask_w {
            let v = match view.ndim() {
                4 => view[[0, 0, y, x]],
                3 => view[[0, y, x]],
                2 => view[[y, x]],
                _ => unreachable!(),
            };
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            raw.push(v);
        }
    }

    let range = (max_v - min_v).max(1e-6f32);
    let mut mask = GrayImage::new(mask_w as u32, mask_h as u32);
    let mut idx = 0usize;
    for y in 0..mask_h {
        for x in 0..mask_w {
            let normalized = ((raw[idx] - min_v) / range).clamp(0.0f32, 1.0f32);
            let alpha = (normalized * 255.0f32) as u8;
            mask.put_pixel(x as u32, y as u32, Luma([alpha]));
            idx += 1;
        }
    }

    let full_size = image::imageops::resize(&mask, orig_w, orig_h, FilterType::Triangle);
    let mut encoded = Vec::new();
    DynamicImage::ImageLuma8(full_size).write_to(&mut std::io::Cursor::new(&mut encoded), ImageFormat::Png)?;
    Ok(encoded)
}
