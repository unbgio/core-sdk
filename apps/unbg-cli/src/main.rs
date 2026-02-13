use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::{Args, Parser, Subcommand};
use image::GenericImageView;
use regex::Regex;
use walkdir::WalkDir;
use unbg_core::{
    run_inference_with_telemetry, ExecutionProvider, GpuBackendPreference, InferenceRequest, ModelKind, OnnxVariant, PlatformTarget,
    RuntimeConfig, RuntimePolicy,
};
use unbg_installer::{install_models, verify_models, InstallRequest};
use unbg_model_registry::{model_revision_dir, read_lockfile, resolve_model_paths, KnownModel};
use unbg_telemetry::sink_from_env;
use unbg_runtime_ort::LocalOrtBackend;

#[derive(Parser, Debug)]
#[command(name = "unbg", version, about = "UNBG local model tooling")]
struct Cli {
    #[command(subcommand)]
    command: TopLevelCommand,
}

#[derive(Subcommand, Debug)]
enum TopLevelCommand {
    Models(ModelsCommand),
    #[command(name = "exec")]
    Exec(ExecArgs),
}

#[derive(Args, Debug)]
struct ModelsCommand {
    #[command(subcommand)]
    command: ModelsSubcommand,
}

#[derive(Subcommand, Debug)]
enum ModelsSubcommand {
    Install(InstallArgs),
    List(CommonModelArgs),
    Verify(CommonModelArgs),
    Update(UpdateArgs),
}

#[derive(Args, Debug)]
struct CommonModelArgs {
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct InstallArgs {
    #[arg(long)]
    all: bool,
    #[arg(long = "model")]
    models: Vec<String>,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "HF_TOKEN")]
    hf_token_env: String,
    #[arg(long, default_value = "main")]
    revision_rmbg14: String,
    #[arg(long, default_value = "main")]
    revision_rmbg20: String,
    #[arg(long)]
    verify_only: bool,
    #[arg(long, default_value = "fp16")]
    onnx_variant: String,
}

#[derive(Args, Debug)]
struct UpdateArgs {
    #[arg(long = "model")]
    models: Vec<String>,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, default_value = "HF_TOKEN")]
    hf_token_env: String,
    #[arg(long, default_value = "fp16")]
    onnx_variant: String,
}

#[derive(Args, Debug)]
struct ExecArgs {
    #[arg(long, short = 'i')]
    input: String,
    /// Root directory for regex input matching (defaults to current directory).
    #[arg(long, short = 'r')]
    input_root: Option<PathBuf>,
    /// Recurse when scanning directories / regex matches.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    recursive: bool,
    /// If set, abort the whole run on the first input error.
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    strict: bool,
    #[arg(long, short = 'M', default_value = "fast")]
    model: String,
    #[arg(long, short = 'p', default_value_t = 2_000_000)]
    max_inference_pixels: u32,
    #[arg(long, short = 'a', default_value_t = true)]
    allow_rmbg20: bool,
    #[arg(long, short = 'd')]
    model_dir: Option<PathBuf>,
    #[arg(long, short = 'o')]
    output_cutout: Option<PathBuf>,
    #[arg(long, short = 'm')]
    output_mask: Option<PathBuf>,
    /// Output directory used when processing multiple inputs.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long, short = 'v', default_value = "fp16")]
    onnx_variant: String,
    #[arg(long, short = 'e', default_value = "gpu")]
    execution_provider: String,
    #[arg(long, short = 'g', default_value = "auto")]
    gpu_backend: String,
    #[arg(long, short = 'b', default_value_t = false, action = clap::ArgAction::Set)]
    benchmark_provider: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    profile: bool,
    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    inference_only: bool,
    #[arg(long, default_value_t = 1)]
    repeat: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        TopLevelCommand::Models(models) => match models.command {
            ModelsSubcommand::Install(args) => {
                let report = install_models(&InstallRequest {
                    model_dir: args.model_dir,
                    install_all: args.all,
                    models: parse_models_for_install(&args.models)?,
                    hf_token_env: args.hf_token_env,
                    revision_rmbg14: args.revision_rmbg14,
                    revision_rmbg20: args.revision_rmbg20,
                    verify_only: args.verify_only,
                    onnx_variant: parse_onnx_variant(&args.onnx_variant)?,
                })?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            ModelsSubcommand::List(args) => {
                let lock = verify_models(args.model_dir)?;
                println!("{}", serde_json::to_string_pretty(&lock.models)?);
            }
            ModelsSubcommand::Verify(args) => {
                let lock = verify_models(args.model_dir)?;
                println!("{}", serde_json::to_string_pretty(&lock)?);
            }
            ModelsSubcommand::Update(args) => {
                let parsed = parse_models_for_install(&args.models)?;
                let report = install_models(&InstallRequest {
                    model_dir: args.model_dir,
                    install_all: parsed.is_empty() || args.models.iter().any(|m| m.eq_ignore_ascii_case("all")),
                    models: parsed,
                    hf_token_env: args.hf_token_env,
                    revision_rmbg14: "main".to_string(),
                    revision_rmbg20: "main".to_string(),
                    verify_only: false,
                    onnx_variant: parse_onnx_variant(&args.onnx_variant)?,
                })?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        },
        TopLevelCommand::Exec(args) => {
            let total_start = Instant::now();
            set_ort_dylib_path_if_available();
            let model_ensure_start = Instant::now();
            let mut timings = serde_json::Map::new();
            timings.insert(
                "setupRuntimePath".to_string(),
                serde_json::json!(model_ensure_start.duration_since(total_start).as_millis()),
            );

            let inputs = resolve_exec_inputs(&args)?;
            if inputs.is_empty() {
                return Err(anyhow!("no input images matched"));
            }
            let runtime_cfg = unbg_core::resolve_runtime_config(RuntimeConfig {
                model: args.model.clone(),
                onnx_variant: args.onnx_variant.clone(),
                execution_provider: args.execution_provider.clone(),
                gpu_backend: args.gpu_backend.clone(),
                benchmark_provider: args.benchmark_provider,
                model_dir: args.model_dir.as_ref().map(|path| path.display().to_string()),
            });
            let requested_model = parse_model_choice(&runtime_cfg.model)?;
            let onnx_variant = parse_onnx_variant(&runtime_cfg.onnx_variant)?;
            ensure_models_for_exec(&args, requested_model, onnx_variant)?;
            let model_ensure_done = Instant::now();
            let policy = RuntimePolicy {
                max_inference_pixels: args.max_inference_pixels,
                max_latency_ms: 1_500,
                allow_rmbg20: args.allow_rmbg20,
            };
            let backend = LocalOrtBackend::default();
            let telemetry = sink_from_env();
            let telemetry_ref = telemetry.as_ref().map(|sink| sink.as_ref());

            let bulk_mode = inputs.len() > 1;
            let mut results = Vec::with_capacity(inputs.len());
            let mut total_inference_ms: u128 = 0;
            let mut total_write_ms: u128 = 0;

            for input_path in inputs {
                let read_start = Instant::now();
                let source = match std::fs::read(&input_path) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        if bulk_mode && !args.strict {
                            results.push(serde_json::json!({
                                "input": input_path,
                                "error": format!("failed to read input: {}", err),
                            }));
                            continue;
                        }
                        return Err(anyhow!("failed to read input {}: {}", input_path.display(), err));
                    }
                };
                let read_done = Instant::now();
                let image = match image::load_from_memory(&source) {
                    Ok(img) => img,
                    Err(err) => {
                        if bulk_mode && !args.strict {
                            results.push(serde_json::json!({
                                "input": input_path,
                                "error": format!("failed to decode input: {}", err),
                            }));
                            continue;
                        }
                        return Err(anyhow!("failed to decode input {}: {}", input_path.display(), err));
                    }
                };
                let decode_done = Instant::now();
                let (width, height) = image.dimensions();

                let (output_cutout, output_mask) = resolve_outputs_for_input(&args, &input_path)?;
                let request = InferenceRequest {
                    requested_model,
                    onnx_variant,
                    execution_provider: parse_execution_provider(&runtime_cfg.execution_provider)?,
                    gpu_backend: parse_gpu_backend(&runtime_cfg.gpu_backend)?,
                    benchmark_provider: runtime_cfg.benchmark_provider,
                    emit_mask_png: !args.inference_only,
                    input_path: Some(input_path.clone()),
                    input_bytes: Some(source.clone()),
                    model_dir: runtime_cfg.model_dir.clone().map(PathBuf::from),
                    width,
                    height,
                };

                let mut last_result = None;
                let inference_start = Instant::now();
                for _ in 0..args.repeat.max(1) {
                    let result = run_inference_with_telemetry(&backend, &request, &policy, PlatformTarget::Cli, telemetry_ref)?;
                    last_result = Some(result);
                }
                let inference_done = Instant::now();
                let result = last_result.ok_or_else(|| anyhow!("inference did not produce a result"))?;
                total_inference_ms += inference_done.duration_since(inference_start).as_millis();

                let write_start = Instant::now();
                if let Some(ref mask_path) = output_mask {
                    if let Some(parent) = mask_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(mask_path, &result.mask_png)?;
                }
                if let Some(ref cutout_path) = output_cutout {
                    write_cutout_png(&source, &result.mask_png, &cutout_path)?;
                }
                let write_done = Instant::now();
                total_write_ms += write_done.duration_since(write_start).as_millis();

                let mut per = serde_json::Map::new();
                if args.profile {
                    per.insert(
                        "readInput".to_string(),
                        serde_json::json!(read_done.duration_since(read_start).as_millis()),
                    );
                    per.insert(
                        "decodeInput".to_string(),
                        serde_json::json!(decode_done.duration_since(read_done).as_millis()),
                    );
                    per.insert(
                        "inference".to_string(),
                        serde_json::json!(inference_done.duration_since(inference_start).as_millis()),
                    );
                    per.insert(
                        "writeOutputs".to_string(),
                        serde_json::json!(write_done.duration_since(write_start).as_millis()),
                    );
                }

                results.push(serde_json::json!({
                    "input": input_path,
                    "modelUsed": model_kind_label(result.model_used),
                    "providerSelected": result.execution_provider_selected,
                    "backendSelected": result.gpu_backend_selected,
                    "fallbackUsed": result.fallback_used,
                    "width": result.width,
                    "height": result.height,
                    "outputMask": output_mask,
                    "outputCutout": output_cutout,
                    "timingsMs": if args.profile { Some(serde_json::Value::Object(per)) } else { None }
                }));
            }

            let done = Instant::now();
            if args.profile {
                timings.insert(
                    "ensureModels".to_string(),
                    serde_json::json!(model_ensure_done.duration_since(model_ensure_start).as_millis()),
                );
                timings.insert("repeat".to_string(), serde_json::json!(args.repeat.max(1)));
                timings.insert("files".to_string(), serde_json::json!(results.len()));
                timings.insert("inference".to_string(), serde_json::json!(total_inference_ms));
                timings.insert("writeOutputs".to_string(), serde_json::json!(total_write_ms));
                timings.insert("total".to_string(), serde_json::json!(done.duration_since(total_start).as_millis()));
            }

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "results": results,
                    "timingsMs": if args.profile { Some(serde_json::Value::Object(timings)) } else { None }
                }))?
            );
        }
    }

    Ok(())
}

fn resolve_exec_inputs(args: &ExecArgs) -> Result<Vec<PathBuf>> {
    let candidate = PathBuf::from(&args.input);
    if candidate.exists() {
        if candidate.is_dir() {
            return collect_images_in_dir(&candidate, args.recursive);
        }
        return Ok(vec![candidate]);
    }
    // Treat as regex matching file name under input_root.
    let root = args
        .input_root
        .clone()
        .unwrap_or(std::env::current_dir().map_err(|e| anyhow!(e.to_string()))?);
    let re = Regex::new(&args.input).map_err(|e| anyhow!("invalid regex: {}", e))?;
    collect_images_by_regex(&root, args.recursive, &re)
}

fn collect_images_in_dir(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let walker = if recursive {
        WalkDir::new(dir)
    } else {
        WalkDir::new(dir).max_depth(1)
    };
    for entry in walker.into_iter().filter_map(std::result::Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.into_path();
        if is_supported_image(&p) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn collect_images_by_regex(root: &Path, recursive: bool, re: &Regex) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let walker = if recursive {
        WalkDir::new(root)
    } else {
        WalkDir::new(root).max_depth(1)
    };
    for entry in walker.into_iter().filter_map(std::result::Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.into_path();
        if !is_supported_image(&p) {
            continue;
        }
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if re.is_match(name) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn is_supported_image(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    matches!(ext.as_str(), "png" | "jpg" | "jpeg")
}

fn resolve_outputs_for_input(args: &ExecArgs, input_path: &Path) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    if args.inference_only {
        return Ok((None, None));
    }

    let multi_input = {
        let as_path = PathBuf::from(&args.input);
        (as_path.exists() && as_path.is_dir()) || !as_path.exists()
    };

    // When multi-input, prefer explicit --output-dir, otherwise interpret -o/-m as directories.
    let bulk_out_dir = if multi_input { args.output_dir.clone() } else { None };

    let cutout = if let Some(spec) = args.output_cutout.clone() {
        if multi_input {
            let dir = bulk_out_dir.unwrap_or(spec);
            Some(dir.join(default_cutout_filename(input_path)?))
        } else {
            validate_cutout_extension(&spec)?;
            Some(spec)
        }
    } else if args.output_mask.is_none() {
        if let Some(dir) = bulk_out_dir {
            Some(dir.join(default_cutout_filename(input_path)?))
        } else {
            Some(default_cutout_path(input_path)?)
        }
    } else {
        None
    };

    let mask = if let Some(spec) = args.output_mask.clone() {
        if multi_input {
            let dir = args.output_dir.clone().unwrap_or(spec);
            Some(dir.join(default_mask_filename(input_path)?))
        } else {
            Some(spec)
        }
    } else {
        None
    };

    Ok((cutout, mask))
}

fn default_cutout_filename(input: &Path) -> Result<String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| anyhow!("input file must include a valid file name"))?
        .to_string_lossy();
    Ok(format!("{}_cutout.png", stem))
}

fn default_mask_filename(input: &Path) -> Result<String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| anyhow!("input file must include a valid file name"))?
        .to_string_lossy();
    Ok(format!("{}_mask.png", stem))
}

fn ensure_models_for_exec(args: &ExecArgs, requested_model: ModelKind, onnx_variant: OnnxVariant) -> Result<()> {
    let required_models: Vec<KnownModel> = match requested_model {
        ModelKind::Rmbg14 | ModelKind::Auto => vec![KnownModel::Rmbg14],
        ModelKind::Rmbg20 => vec![KnownModel::Rmbg20],
    };
    let missing_any = !has_required_models_for_exec(args.model_dir.as_deref(), &required_models)?;
    if !missing_any {
        return Ok(());
    }
    eprintln!("Installing required models before execution...");
    let report = install_models(&InstallRequest {
        model_dir: args.model_dir.clone(),
        install_all: false,
        models: required_models,
        hf_token_env: "HF_TOKEN".to_string(),
        revision_rmbg14: "main".to_string(),
        revision_rmbg20: "main".to_string(),
        verify_only: false,
        onnx_variant,
    })?;
    if report.installed.is_empty() && report.skipped.is_empty() {
        eprintln!("Model install step completed.");
    }
    Ok(())
}

fn has_required_models_for_exec(model_dir: Option<&Path>, required_models: &[KnownModel]) -> Result<bool> {
    let paths = resolve_model_paths(model_dir)?;
    let lock = match read_lockfile(&paths) {
        Ok(lock) => lock,
        Err(_) => return Ok(false),
    };
    for model in required_models {
        let revision = "main";
        let has_entry = lock
            .models
            .iter()
            .any(|entry| entry.model_id == model.model_id() && entry.revision == revision);
        if !has_entry {
            return Ok(false);
        }
        let rev_dir = model_revision_dir(&paths, *model, revision);
        if !directory_has_onnx_file(&rev_dir) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn directory_has_onnx_file(dir: &Path) -> bool {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .any(|entry| {
            entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("onnx"))
                    .unwrap_or(false)
        })
}

fn default_cutout_path(input: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .ok_or_else(|| anyhow!("input file must include a valid file name"))?
        .to_string_lossy();
    let filename = format!("{}_cutout.png", stem);
    let out_path = if let Some(parent) = input.parent() {
        parent.join(filename)
    } else {
        PathBuf::from(filename)
    };
    Ok(out_path)
}

fn validate_cutout_extension(path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext != "png" {
        return Err(anyhow!(
            "output cutout must be a .png file (received: '{}')",
            path.display()
        ));
    }
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
    if let Some(exe_candidate) = discover_ort_next_to_executable(lib_name) {
        std::env::set_var("ORT_DYLIB_PATH", exe_candidate);
        return;
    }
    if let Some(python_candidate) = discover_ort_from_python(lib_name) {
        std::env::set_var("ORT_DYLIB_PATH", python_candidate);
        return;
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            if cfg!(target_os = "windows") && dir.to_string_lossy().to_ascii_lowercase().contains("windows\\system32") {
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

fn discover_ort_next_to_executable(lib_name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join(lib_name);
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

fn discover_ort_from_python(lib_name: &str) -> Option<PathBuf> {
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
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn parse_models_for_install(models: &[String]) -> Result<Vec<KnownModel>> {
    let mut out = Vec::with_capacity(models.len());
    for model in models {
        let normalized = model.to_ascii_lowercase();
        let parsed = match normalized.as_str() {
            "rmbg-1.4" | "fast" => KnownModel::Rmbg14,
            "rmbg-2.0" | "quality" => KnownModel::Rmbg20,
            "all" => continue,
            other => {
                return Err(anyhow!(
                    "unknown model '{}'; expected one of: rmbg-1.4 (fast), rmbg-2.0 (quality), all",
                    other
                ));
            }
        };
        out.push(parsed);
    }
    Ok(out)
}

fn parse_model_choice(model: &str) -> Result<ModelKind> {
    match model.to_ascii_lowercase().as_str() {
        "auto" => Ok(ModelKind::Auto),
        "rmbg-1.4" | "fast" => Ok(ModelKind::Rmbg14),
        "rmbg-2.0" | "quality" => Ok(ModelKind::Rmbg20),
        other => Err(anyhow!(
            "unknown model '{}'; expected one of: auto, rmbg-1.4 (fast), rmbg-2.0 (quality)",
            other
        )),
    }
}

fn model_kind_label(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Auto => "auto",
        ModelKind::Rmbg14 => "rmbg-1.4",
        ModelKind::Rmbg20 => "rmbg-2.0",
    }
}

fn parse_onnx_variant(value: &str) -> Result<OnnxVariant> {
    match value.to_ascii_lowercase().as_str() {
        "fp16" => Ok(OnnxVariant::Fp16),
        "fp32" => Ok(OnnxVariant::Fp32),
        "quantized" | "q8" => Ok(OnnxVariant::Quantized),
        "auto" => Ok(OnnxVariant::Auto),
        other => Err(anyhow!(
            "unknown onnx variant '{}'; expected one of: fp16, fp32, quantized, auto",
            other
        )),
    }
}

fn parse_execution_provider(value: &str) -> Result<ExecutionProvider> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(ExecutionProvider::Auto),
        "gpu" => Ok(ExecutionProvider::Gpu),
        "cpu" => Ok(ExecutionProvider::Cpu),
        other => Err(anyhow!(
            "unknown execution provider '{}'; expected one of: auto, gpu, cpu",
            other
        )),
    }
}

fn parse_gpu_backend(value: &str) -> Result<GpuBackendPreference> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(GpuBackendPreference::Auto),
        "directml" => Ok(GpuBackendPreference::DirectML),
        "cuda" => Ok(GpuBackendPreference::Cuda),
        "coreml" => Ok(GpuBackendPreference::CoreML),
        "metal" => Ok(GpuBackendPreference::Metal),
        other => Err(anyhow!(
            "unknown gpu backend '{}'; expected one of: auto, directml, cuda, coreml, metal",
            other
        )),
    }
}

fn write_cutout_png(source_bytes: &[u8], mask_png: &[u8], out_path: &std::path::Path) -> Result<()> {
    let source = image::load_from_memory(source_bytes)?.to_rgba8();
    let mask = image::load_from_memory(mask_png)?.to_luma8();
    let (w, h) = source.dimensions();
    if mask.dimensions() != (w, h) {
        return Err(anyhow!("mask dimensions do not match source dimensions"));
    }

    let mut cutout = source.clone();
    for y in 0..h {
        for x in 0..w {
            let alpha = mask.get_pixel(x, y)[0];
            let px = cutout.get_pixel_mut(x, y);
            px[3] = alpha;
        }
    }
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    cutout.save(out_path)?;
    Ok(())
}
