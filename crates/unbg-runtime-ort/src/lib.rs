use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{
    imageops::FilterType, DynamicImage, ExtendedColorType, GrayImage, ImageBuffer, ImageEncoder,
    ImageFormat, ImageReader, Luma,
};
use ort::{inputs, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use unbg_core::{
    CoreError, ExecutionProvider, GpuBackendPreference, InferenceBackend, InferenceRequest,
    InferenceResult, ModelKind, OnnxVariant,
};
use unbg_model_registry::{
    model_revision_dir, read_lockfile, resolve_model_paths, safe_join, KnownModel, LockFileEntry,
};

pub const MAX_ENCODED_INPUT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_DECODED_INPUT_DIMENSION: u32 = 16_384;
pub const MAX_DECODED_INPUT_PIXELS: u64 = 40_000_000;
const MAX_DECODE_ALLOC_BYTES: u64 = 256 * 1024 * 1024;
const MAX_MODEL_OUTPUT_PIXELS: usize = 16 * 1024 * 1024;
const MAX_PROVIDER_CACHE_BYTES: u64 = 64 * 1024;
const MAX_PROVIDER_CACHE_ENTRIES: usize = 16;

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

static AUTO_PROVIDER_CACHE: OnceLock<Mutex<std::collections::HashMap<String, ProviderChoice>>> =
    OnceLock::new();

struct CachedSession {
    key: String,
    session: Session,
}

// A process-wide cache deliberately holds at most one ORT session. A thread-local
// cache lets every worker retain a full model, which can exhaust memory under
// concurrent FFI or Tauri calls.
static SESSION_CACHE: OnceLock<Mutex<Option<CachedSession>>> = OnceLock::new();

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
            return decode_image(bytes);
        }
        if let Some(path) = &request.input_path {
            let bytes = read_bounded_input(path)?;
            return decode_image(&bytes);
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

fn read_bounded_input(path: &Path) -> Result<Vec<u8>, CoreError> {
    let encoded_size = fs::metadata(path)
        .map_err(|error| CoreError::Backend(error.to_string()))?
        .len();
    validate_encoded_input_size(encoded_size)?;
    let file = fs::File::open(path).map_err(|error| CoreError::Backend(error.to_string()))?;
    let mut bytes = Vec::with_capacity(usize::try_from(encoded_size).unwrap_or(0));
    file.take(MAX_ENCODED_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| CoreError::Backend(error.to_string()))?;
    validate_encoded_input_size(bytes.len() as u64)?;
    Ok(bytes)
}

impl InferenceBackend for LocalOrtBackend {
    fn infer(
        &self,
        request: &InferenceRequest,
        selected_model: ModelKind,
    ) -> Result<InferenceResult, CoreError> {
        let image = match self.load_image(request) {
            Ok(img) => img,
            Err(err) => {
                if placeholder_fallback_allowed() {
                    let width = request.width.max(1);
                    let height = request.height.max(1);
                    validate_decoded_input_dimensions(width, height)?;
                    return self
                        .infer_fallback(selected_model, DynamicImage::new_rgb8(width, height));
                }
                return Err(err);
            }
        };
        if request.width != image.width() || request.height != image.height() {
            return Err(backend_error(
                "input-metadata",
                format!(
                    "declared image dimensions {}x{} do not match decoded dimensions {}x{}",
                    request.width,
                    request.height,
                    image.width(),
                    image.height()
                ),
            ));
        }

        let variants_to_try = onnx_variants_to_try(request.onnx_variant);
        let candidates = candidate_providers(request);
        if candidates.is_empty() {
            return Err(CoreError::Backend(
                "no execution providers available".to_string(),
            ));
        }

        let mut last_err = None;
        for variant in &variants_to_try {
            let mut variant_request = request.clone();
            variant_request.onnx_variant = *variant;

            let model_file = match resolve_model_onnx_file(&variant_request, selected_model) {
                Ok(path) => path,
                Err(err) => {
                    last_err = Some(err);
                    continue;
                }
            };

            let result = if request.execution_provider == ExecutionProvider::Auto {
                // `benchmark_provider` remains in the public request for API
                // compatibility, but multi-provider benchmarking is disabled.
                // ORT can retain model allocations after a Session is dropped,
                // so probing several providers can exhaust process memory.
                run_auto_cached_path(
                    &image,
                    &model_file,
                    selected_model,
                    &variant_request,
                    &candidates,
                )
            } else {
                run_sequential_path(
                    &image,
                    &model_file,
                    selected_model,
                    &candidates,
                    request.emit_mask_png,
                )
            };

            match result {
                Ok(res) => return Ok(res),
                Err(err) => {
                    last_err = Some(err);
                    clear_session_cache();
                    continue;
                }
            }
        }

        let err = last_err
            .unwrap_or_else(|| CoreError::Backend("no onnx variants available".to_string()));
        if placeholder_fallback_allowed() {
            self.infer_fallback(selected_model, image)
        } else {
            Err(err)
        }
    }
}

fn decode_image(bytes: &[u8]) -> Result<DynamicImage, CoreError> {
    validate_encoded_input_size(bytes.len() as u64)?;

    let dimensions_reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| CoreError::Backend(format!("image format error: {e}")))?;
    let (width, height) = dimensions_reader
        .into_dimensions()
        .map_err(|e| CoreError::Backend(format!("image header error: {e}")))?;
    validate_decoded_input_dimensions(width, height)?;

    let mut decode_reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| CoreError::Backend(format!("image format error: {e}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODED_INPUT_DIMENSION);
    limits.max_image_height = Some(MAX_DECODED_INPUT_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_ALLOC_BYTES);
    decode_reader.limits(limits);
    let image = decode_reader
        .decode()
        .map_err(|e| CoreError::Backend(format!("image decode error: {e}")))?;
    validate_decoded_input_dimensions(image.width(), image.height())?;
    Ok(image)
}

fn validate_encoded_input_size(size: u64) -> Result<(), CoreError> {
    if size > MAX_ENCODED_INPUT_BYTES {
        return Err(backend_error(
            "input-limit",
            format!("encoded image is {size} bytes; maximum is {MAX_ENCODED_INPUT_BYTES} bytes"),
        ));
    }
    Ok(())
}

fn validate_decoded_input_dimensions(width: u32, height: u32) -> Result<(), CoreError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            backend_error(
                "input-limit",
                "decoded image dimensions overflow".to_string(),
            )
        })?;
    if width == 0 || height == 0 {
        return Err(backend_error(
            "input-limit",
            "decoded image dimensions must be non-zero".to_string(),
        ));
    }
    if width > MAX_DECODED_INPUT_DIMENSION || height > MAX_DECODED_INPUT_DIMENSION {
        return Err(backend_error(
            "input-limit",
            format!(
                "decoded image is {width}x{height}; maximum dimension is {MAX_DECODED_INPUT_DIMENSION}"
            ),
        ));
    }
    if pixels > MAX_DECODED_INPUT_PIXELS {
        return Err(backend_error(
            "input-limit",
            format!(
                "decoded image has {pixels} pixels; maximum is {MAX_DECODED_INPUT_PIXELS} pixels"
            ),
        ));
    }
    Ok(())
}

fn onnx_variants_to_try(requested: OnnxVariant) -> Vec<OnnxVariant> {
    match requested {
        OnnxVariant::Fp16 => vec![OnnxVariant::Fp16, OnnxVariant::Fp32],
        OnnxVariant::Fp32 => vec![OnnxVariant::Fp32, OnnxVariant::Fp16],
        OnnxVariant::Quantized => {
            vec![OnnxVariant::Quantized, OnnxVariant::Fp16, OnnxVariant::Fp32]
        }
        OnnxVariant::Auto => vec![OnnxVariant::Auto, OnnxVariant::Fp32],
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

fn resolve_model_onnx_file(
    request: &InferenceRequest,
    selected_model: ModelKind,
) -> Result<PathBuf, CoreError> {
    let paths = resolve_model_paths(request.model_dir.as_deref())
        .map_err(|e| CoreError::Backend(e.to_string()))?;
    let lock = read_lockfile(&paths).map_err(|e| CoreError::Backend(e.to_string()))?;
    let wanted_id = match selected_model {
        ModelKind::Rmbg14 => KnownModel::Rmbg14.model_id(),
        ModelKind::Rmbg20 => KnownModel::Rmbg20.model_id(),
        ModelKind::Auto => {
            return Err(CoreError::Backend(
                "auto model cannot resolve onnx directly".to_string(),
            ))
        }
    };
    let model = lock
        .models
        .iter()
        .find(|m| m.model_id == wanted_id)
        .ok_or_else(|| CoreError::Backend(format!("model not found in lockfile: {}", wanted_id)))?;
    let known_model = KnownModel::from_model_id(&model.model_id)
        .ok_or_else(|| CoreError::Backend(format!("unknown model id: {}", model.model_id)))?;
    let rev_dir = model_revision_dir(&paths, known_model, &model.revision)
        .map_err(|e| CoreError::Backend(e.to_string()))?;
    find_preferred_onnx_file(&rev_dir, &model.files, request.onnx_variant)
        .map_err(|e| CoreError::Backend(e.to_string()))?
        .ok_or_else(|| {
            CoreError::Backend(format!(
                "no tracked .onnx file found for {} revision {} in {}",
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

fn run_auto_cached_path(
    image: &DynamicImage,
    model_file: &Path,
    selected_model: ModelKind,
    request: &InferenceRequest,
    candidates: &[ProviderChoice],
) -> Result<InferenceResult, CoreError> {
    let cache_key = provider_cache_key(selected_model, request);
    let mut failed_cached_provider = None;
    if let Some(cached) = load_cached_provider(&cache_key, request.model_dir.as_deref()) {
        if candidates.contains(&cached) {
            if let Ok((result, _)) = run_provider(
                image,
                model_file,
                selected_model,
                cached,
                request.emit_mask_png,
            ) {
                return Ok(result);
            }
            failed_cached_provider = Some(cached);
            clear_session_cache();
        }
    }

    let mut errors = Vec::new();
    for provider in candidates {
        if Some(*provider) == failed_cached_provider {
            continue;
        }
        match run_provider(
            image,
            model_file,
            selected_model,
            *provider,
            request.emit_mask_png,
        ) {
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
    let cache = SESSION_CACHE.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .map_err(|_| anyhow!("session cache lock poisoned"))?;
    if cached.as_ref().map(|entry| entry.key.as_str()) != Some(session_key.as_str()) {
        // Drop the previous model before constructing its replacement. This is
        // important for large RMBG-2.0 sessions and provider fallback.
        *cached = None;
        let session = build_session_for_provider(model_file, provider)?;
        *cached = Some(CachedSession {
            key: session_key,
            session,
        });
    }
    let session = &mut cached
        .as_mut()
        .ok_or_else(|| anyhow!("session cache failed to initialize"))?
        .session;
    let mask_png = run_onnx_inference(image, session, selected_model, emit_mask_png)?;
    let elapsed = start.elapsed().as_millis();
    Ok((
        inference_result(image, selected_model, provider, mask_png),
        elapsed,
    ))
}

fn inference_result(
    image: &DynamicImage,
    selected_model: ModelKind,
    provider: ProviderChoice,
    mask_png: Vec<u8>,
) -> InferenceResult {
    let (execution_provider_selected, gpu_backend_selected) = match provider {
        ProviderChoice::Cpu => ("cpu".to_string(), None),
        ProviderChoice::DirectML => ("gpu".to_string(), Some("directml".to_string())),
        ProviderChoice::Cuda => ("gpu".to_string(), Some("cuda".to_string())),
        ProviderChoice::CoreML => ("gpu".to_string(), Some("coreml".to_string())),
    };
    InferenceResult {
        model_used: selected_model,
        mask_png,
        width: image.width(),
        height: image.height(),
        execution_provider_selected,
        gpu_backend_selected,
        fallback_used: false,
    }
}

fn clear_session_cache() {
    if let Some(cache) = SESSION_CACHE.get() {
        if let Ok(mut cached) = cache.lock() {
            *cached = None;
        }
    }
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
    let memory_cache =
        AUTO_PROVIDER_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Ok(cache) = memory_cache.lock() {
        if let Some(provider) = cache.get(cache_key).copied() {
            return Some(provider);
        }
    }

    let cache_path = provider_cache_file(model_dir)?;
    let parsed = read_provider_cache(&cache_path)?;
    let provider = parsed
        .providers
        .get(cache_key)
        .and_then(|v| parse_provider_choice(v))?;
    cache_provider_choice(memory_cache, cache_key, provider);
    Some(provider)
}

fn persist_cached_provider(cache_key: &str, provider: ProviderChoice, model_dir: Option<&Path>) {
    let memory_cache =
        AUTO_PROVIDER_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    cache_provider_choice(memory_cache, cache_key, provider);

    let Some(cache_path) = provider_cache_file(model_dir) else {
        return;
    };
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if fs::symlink_metadata(&cache_path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return;
    }
    let mut updated = read_provider_cache(&cache_path).unwrap_or_default();
    if updated.providers.len() >= MAX_PROVIDER_CACHE_ENTRIES
        && !updated.providers.contains_key(cache_key)
    {
        updated.providers.clear();
    }
    updated
        .providers
        .insert(cache_key.to_string(), provider_label(provider).to_string());
    if let Ok(serialized) = serde_json::to_string_pretty(&updated) {
        if serialized.len() as u64 > MAX_PROVIDER_CACHE_BYTES {
            return;
        }
        let _ = fs::write(cache_path, serialized);
    }
}

fn read_provider_cache(path: &Path) -> Option<PersistedProviderCache> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if metadata.file_type().is_symlink() || metadata.len() > MAX_PROVIDER_CACHE_BYTES {
        return None;
    }
    let raw = fs::read(path).ok()?;
    if raw.len() as u64 > MAX_PROVIDER_CACHE_BYTES {
        return None;
    }
    serde_json::from_slice(&raw).ok()
}

fn cache_provider_choice(
    cache: &Mutex<std::collections::HashMap<String, ProviderChoice>>,
    cache_key: &str,
    provider: ProviderChoice,
) {
    if let Ok(mut entries) = cache.lock() {
        if entries.len() >= MAX_PROVIDER_CACHE_ENTRIES && !entries.contains_key(cache_key) {
            entries.clear();
        }
        entries.insert(cache_key.to_string(), provider);
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
    match request.execution_provider {
        ExecutionProvider::Cpu => vec![ProviderChoice::Cpu],
        ExecutionProvider::Auto | ExecutionProvider::Gpu => {
            let mut providers: Vec<_> = gpu_candidates(request.gpu_backend)
                .into_iter()
                .take(1)
                .collect();
            providers.push(ProviderChoice::Cpu);
            providers
        }
    }
}

fn gpu_candidates(pref: GpuBackendPreference) -> Vec<ProviderChoice> {
    let mut providers = Vec::new();
    match pref {
        GpuBackendPreference::DirectML => providers.push(ProviderChoice::DirectML),
        GpuBackendPreference::Cuda => providers.push(ProviderChoice::Cuda),
        GpuBackendPreference::CoreML | GpuBackendPreference::Metal => {
            providers.push(ProviderChoice::CoreML)
        }
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
        false
    }
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
        ];
        candidates.iter().any(|p| Path::new(p).exists())
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

fn provider_label(provider: ProviderChoice) -> &'static str {
    match provider {
        ProviderChoice::Cpu => "cpu",
        ProviderChoice::DirectML => "directml",
        ProviderChoice::Cuda => "cuda",
        ProviderChoice::CoreML => "coreml",
    }
}

fn is_fp16_model(model_file: &Path) -> bool {
    model_file.to_string_lossy().to_lowercase().contains("fp16")
}

fn session_builder_for_model(model_file: &Path) -> Result<ort::session::builder::SessionBuilder> {
    let builder = Session::builder().map_err(|error| anyhow!(error.to_string()))?;
    if is_fp16_model(model_file) {
        // RMBG-2.0 fp16 ONNX export contains InsertedPrecisionFreeCast nodes
        // that break ORT's SimplifiedLayerNormFusion optimizer. Disable it.
        Ok(builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
            .map_err(|error| anyhow!(error.to_string()))?)
    } else {
        Ok(builder)
    }
}

fn build_session_for_provider(model_file: &Path, provider: ProviderChoice) -> Result<Session> {
    match provider {
        ProviderChoice::Cpu => session_builder_for_model(model_file)?
            .commit_from_file(model_file)
            .map_err(Into::into),
        ProviderChoice::DirectML => {
            #[cfg(feature = "directml")]
            {
                session_builder_for_model(model_file)?
                    .with_execution_providers([ort::ep::DirectML::default().build()])
                    .map_err(|error| anyhow!(error.to_string()))?
                    .commit_from_file(model_file)
                    .map_err(|error| anyhow!(error.to_string()))
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(anyhow!("directml feature not enabled"))
            }
        }
        ProviderChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                session_builder_for_model(model_file)?
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(|error| anyhow!(error.to_string()))?
                    .commit_from_file(model_file)
                    .map_err(|error| anyhow!(error.to_string()))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!("cuda feature not enabled"))
            }
        }
        ProviderChoice::CoreML => {
            #[cfg(feature = "coreml")]
            {
                session_builder_for_model(model_file)?
                    .with_execution_providers([ort::ep::CoreML::default().build()])
                    .map_err(|error| anyhow!(error.to_string()))?
                    .commit_from_file(model_file)
                    .map_err(|error| anyhow!(error.to_string()))
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

fn find_preferred_onnx_file(
    base_dir: &Path,
    tracked_files: &[LockFileEntry],
    onnx_variant: OnnxVariant,
) -> Result<Option<PathBuf>, unbg_model_registry::RegistryError> {
    let mut candidates = Vec::new();
    for entry in tracked_files {
        if !entry.path.to_ascii_lowercase().ends_with(".onnx") {
            continue;
        }
        let path = safe_join(base_dir, &entry.path, "model lockfile entry")?;
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if metadata.is_file() && metadata.len() == entry.size {
            candidates.push(path);
        }
    }
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
                if lower.contains("model.onnx")
                    && !lower.contains("fp16")
                    && !lower.contains("quantized")
                {
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
    Ok(candidates.into_iter().next())
}

fn run_onnx_inference(
    image: &DynamicImage,
    session: &mut Session,
    selected_model: ModelKind,
    emit_mask_png: bool,
) -> Result<Vec<u8>> {
    let preprocessing = known_model_for_kind(selected_model)?.preprocessing();
    if preprocessing.input_width != preprocessing.input_height {
        return Err(anyhow!(
            "non-square model input is not supported: {}x{}",
            preprocessing.input_width,
            preprocessing.input_height
        ));
    }
    run_onnx_inference_at_size(
        image,
        session,
        selected_model,
        emit_mask_png,
        preprocessing.input_width,
    )
}

fn known_model_for_kind(selected_model: ModelKind) -> Result<KnownModel> {
    match selected_model {
        ModelKind::Rmbg14 => Ok(KnownModel::Rmbg14),
        ModelKind::Rmbg20 => Ok(KnownModel::Rmbg20),
        ModelKind::Auto => Err(anyhow!("auto model must be resolved before preprocessing")),
    }
}

fn prepare_model_input(
    image: &DynamicImage,
    selected_model: ModelKind,
    input_size: u32,
) -> Result<Vec<f32>> {
    let preprocessing = known_model_for_kind(selected_model)?.preprocessing();
    let resized = image
        .resize_exact(input_size, input_size, FilterType::Triangle)
        .to_rgb8();
    let plane_size = input_size as usize * input_size as usize;
    let mut input_data = vec![0f32; 3 * plane_size];
    for y in 0..input_size as usize {
        for x in 0..input_size as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let index = y * input_size as usize + x;
            for channel in 0..3 {
                let scaled = pixel[channel] as f32 / 255.0;
                input_data[channel * plane_size + index] =
                    (scaled - preprocessing.mean[channel]) / preprocessing.std[channel];
            }
        }
    }
    Ok(input_data)
}

fn run_onnx_inference_at_size(
    image: &DynamicImage,
    session: &mut Session,
    selected_model: ModelKind,
    emit_mask_png: bool,
    input_size: u32,
) -> Result<Vec<u8>> {
    let orig_w = image.width();
    let orig_h = image.height();
    let input_data = prepare_model_input(image, selected_model, input_size)?;

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

    // RMBG-1.4 exposes its primary mask first; RMBG-2.0/BiRefNet's
    // reference implementation consumes the last prediction and applies a
    // sigmoid before resizing it to the source image.
    let output_index = if selected_model == ModelKind::Rmbg20 {
        outputs.len() - 1
    } else {
        0
    };
    let view = outputs[output_index].try_extract_array::<f32>()?;

    let (mask_h, mask_w) = match view.ndim() {
        4 if view.shape()[0] > 0 && view.shape()[1] > 0 => (view.shape()[2], view.shape()[3]),
        3 if view.shape()[0] > 0 => (view.shape()[1], view.shape()[2]),
        2 => (view.shape()[0], view.shape()[1]),
        _ => return Err(anyhow!("unsupported output dimensions: {:?}", view.shape())),
    };
    let mask_pixels = mask_w
        .checked_mul(mask_h)
        .ok_or_else(|| anyhow!("model output dimensions overflow"))?;
    if mask_pixels == 0 || mask_pixels > MAX_MODEL_OUTPUT_PIXELS {
        return Err(anyhow!(
            "model output contains {mask_pixels} pixels; maximum is {MAX_MODEL_OUTPUT_PIXELS}"
        ));
    }

    let raw = if let Some(slice) = view.as_slice() {
        if slice.len() < mask_pixels {
            return Err(anyhow!("model output is shorter than its declared shape"));
        }
        slice[..mask_pixels].to_vec()
    } else {
        let mut raw = Vec::with_capacity(mask_pixels);
        for y in 0..mask_h {
            for x in 0..mask_w {
                raw.push(match view.ndim() {
                    4 => view[[0, 0, y, x]],
                    3 => view[[0, y, x]],
                    2 => view[[y, x]],
                    _ => unreachable!(),
                });
            }
        }
        raw
    };

    let mask_w = u32::try_from(mask_w).map_err(|_| anyhow!("model output width is too large"))?;
    let mask_h = u32::try_from(mask_h).map_err(|_| anyhow!("model output height is too large"))?;
    let alpha = postprocess_model_output(raw, mask_w, mask_h, orig_w, orig_h, selected_model)?;
    encode_alpha_mask(&alpha, orig_w, orig_h)
}

fn postprocess_model_output(
    mut raw: Vec<f32>,
    mask_width: u32,
    mask_height: u32,
    output_width: u32,
    output_height: u32,
    selected_model: ModelKind,
) -> Result<Vec<u8>> {
    if selected_model == ModelKind::Rmbg20 {
        for value in &mut raw {
            *value = stable_sigmoid(*value);
        }
    }

    let mask: ImageBuffer<Luma<f32>, Vec<f32>> =
        ImageBuffer::from_vec(mask_width, mask_height, raw)
            .ok_or_else(|| anyhow!("mask buffer build failed"))?;
    let resized = if mask_width == output_width && mask_height == output_height {
        mask
    } else {
        image::imageops::resize(&mask, output_width, output_height, FilterType::Triangle)
    };
    let mut values = resized.into_raw();

    if selected_model == ModelKind::Rmbg14 {
        // The RMBG-1.4 reference implementation resizes logits first, then
        // performs per-image min/max normalization.
        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        for &value in &values {
            if value.is_finite() {
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
        }
        let range = max_value - min_value;
        if !range.is_finite() || range <= f32::EPSILON {
            values.fill(0.0);
        } else {
            for value in &mut values {
                *value = if value.is_finite() {
                    ((*value - min_value) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
        }
    }

    Ok(values
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0) as u8)
        .collect())
}

fn stable_sigmoid(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn encode_alpha_mask(alpha: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let expected = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| anyhow!("mask dimensions are too large"))?;
    if alpha.len() != expected {
        return Err(anyhow!("mask length does not match output dimensions"));
    }

    // LA8 preserves the grayscale mask and mirrors it into alpha, allowing
    // canvas consumers to use destination-in without a pixel readback.
    let mut la = Vec::with_capacity(
        expected
            .checked_mul(2)
            .ok_or_else(|| anyhow!("mask allocation size overflow"))?,
    );
    for &value in alpha {
        la.push(value);
        la.push(value);
    }

    let mut encoded = Vec::new();
    let encoder =
        PngEncoder::new_with_quality(&mut encoded, CompressionType::Fast, PngFilterType::NoFilter);
    encoder.write_image(&la, width, height, ExtendedColorType::La8)?;
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn single_pixel_image(pixel: [u8; 3]) -> DynamicImage {
        let mut image = RgbImage::new(1, 1);
        image.put_pixel(0, 0, Rgb(pixel));
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn configured_ort_runtime_can_initialize() {
        let Some(runtime_path) = std::env::var_os("ORT_DYLIB_PATH") else {
            return;
        };
        assert!(
            Path::new(&runtime_path).is_file(),
            "ORT_DYLIB_PATH must point to a runtime library"
        );
        Session::builder().expect("configured ONNX Runtime must expose the requested API");
    }

    #[test]
    fn rmbg14_preprocessing_matches_reference_values() {
        let input = prepare_model_input(&single_pixel_image([0, 127, 255]), ModelKind::Rmbg14, 1)
            .expect("preprocessing should succeed");

        assert_eq!(input.len(), 3);
        assert_close(input[0], -0.5);
        assert_close(input[1], -0.001_960_784_3);
        assert_close(input[2], 0.5);
    }

    #[test]
    fn rmbg20_preprocessing_uses_imagenet_normalization() {
        let input = prepare_model_input(&single_pixel_image([0, 0, 0]), ModelKind::Rmbg20, 1)
            .expect("preprocessing should succeed");

        assert_eq!(input.len(), 3);
        assert_close(input[0], -2.117_904);
        assert_close(input[1], -2.035_714_1);
        assert_close(input[2], -1.804_444_4);
    }

    #[test]
    fn unresolved_model_cannot_be_preprocessed() {
        let error = prepare_model_input(&single_pixel_image([0, 0, 0]), ModelKind::Auto, 1)
            .expect_err("auto model should be resolved first");
        assert!(error.to_string().contains("must be resolved"));
    }

    #[test]
    fn encoded_input_limit_is_enforced() {
        assert!(validate_encoded_input_size(MAX_ENCODED_INPUT_BYTES).is_ok());
        assert!(validate_encoded_input_size(MAX_ENCODED_INPUT_BYTES + 1).is_err());
    }

    #[test]
    fn decoded_dimension_and_pixel_limits_are_enforced() {
        assert!(validate_decoded_input_dimensions(6_000, 6_000).is_ok());
        assert!(validate_decoded_input_dimensions(0, 1).is_err());
        assert!(validate_decoded_input_dimensions(MAX_DECODED_INPUT_DIMENSION + 1, 1).is_err());
        assert!(validate_decoded_input_dimensions(8_001, 5_000).is_err());
    }

    #[test]
    fn valid_image_is_header_checked_then_decoded() {
        let source = single_pixel_image([10, 20, 30]);
        let mut encoded = Vec::new();
        source
            .write_to(&mut Cursor::new(&mut encoded), ImageFormat::Png)
            .expect("test image should encode");

        let decoded = decode_image(&encoded).expect("valid image should decode");
        assert_eq!((decoded.width(), decoded.height()), (1, 1));
    }

    #[test]
    fn automatic_provider_selection_is_bounded_and_has_cpu_fallback() {
        let request = InferenceRequest {
            requested_model: ModelKind::Rmbg14,
            onnx_variant: OnnxVariant::Fp32,
            execution_provider: ExecutionProvider::Auto,
            gpu_backend: GpuBackendPreference::Auto,
            benchmark_provider: false,
            emit_mask_png: true,
            input_path: None,
            input_bytes: Some(Vec::new()),
            model_dir: None,
            width: 1,
            height: 1,
        };

        let providers = candidate_providers(&request);
        assert!(!providers.is_empty());
        assert!(providers.len() <= 2);
        assert_eq!(providers.last(), Some(&ProviderChoice::Cpu));
    }

    #[test]
    fn explicit_gpu_selection_tries_one_backend_then_cpu() {
        let request = InferenceRequest {
            requested_model: ModelKind::Rmbg14,
            onnx_variant: OnnxVariant::Fp32,
            execution_provider: ExecutionProvider::Gpu,
            gpu_backend: GpuBackendPreference::DirectML,
            benchmark_provider: false,
            emit_mask_png: true,
            input_path: None,
            input_bytes: Some(Vec::new()),
            model_dir: None,
            width: 1,
            height: 1,
        };

        assert_eq!(
            candidate_providers(&request),
            vec![ProviderChoice::DirectML, ProviderChoice::Cpu]
        );
    }

    #[test]
    fn rmbg14_postprocessing_resizes_then_minmax_normalizes() {
        let alpha = postprocess_model_output(vec![2.0, 4.0], 2, 1, 2, 1, ModelKind::Rmbg14)
            .expect("postprocessing should succeed");
        assert_eq!(alpha, vec![0, 255]);
    }

    #[test]
    fn rmbg20_postprocessing_applies_sigmoid_without_minmax() {
        let alpha =
            postprocess_model_output(vec![0.0, 3.0_f32.ln()], 2, 1, 2, 1, ModelKind::Rmbg20)
                .expect("postprocessing should succeed");
        assert_eq!(alpha, vec![127, 191]);
    }

    #[test]
    fn encoded_mask_copies_grayscale_into_alpha() {
        let encoded = encode_alpha_mask(&[0, 255], 2, 1).expect("mask should encode");
        let decoded = image::load_from_memory(&encoded)
            .expect("mask should decode")
            .to_luma_alpha8();
        assert_eq!(decoded.into_raw(), vec![0, 0, 255, 255]);
    }
}
