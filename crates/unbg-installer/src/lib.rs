use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, RANGE, USER_AGENT};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::Builder;
use unbg_core::OnnxVariant;
use unbg_model_registry::{
    built_in_manifest, ensure_layout, merge_lock_models, model_revision_dir, read_lockfile,
    resolve_model_paths, safe_join, write_lockfile, KnownModel, LockFileEntry, LockModel,
    ModelLock,
};
use walkdir::WalkDir;

const MAX_MODEL_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_TREE_RESPONSE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRequest {
    pub model_dir: Option<PathBuf>,
    pub install_all: bool,
    pub models: Vec<KnownModel>,
    pub hf_token_env: String,
    pub revision_rmbg14: String,
    pub revision_rmbg20: String,
    pub verify_only: bool,
    pub onnx_variant: OnnxVariant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallReport {
    pub model_dir: PathBuf,
    pub installed: Vec<String>,
    pub skipped: Vec<String>,
    pub lockfile_written: bool,
}

pub fn install_models(request: &InstallRequest) -> Result<InstallReport> {
    let paths = resolve_model_paths(request.model_dir.as_deref())?;
    ensure_layout(&paths)?;

    let mut targets = request.models.clone();
    if request.install_all || targets.is_empty() {
        targets = KnownModel::all().to_vec();
    }

    let manifest = built_in_manifest();
    let manifest_by_id: HashMap<_, _> = manifest
        .into_iter()
        .map(|m| (m.model_id.clone(), m))
        .collect();
    let token = env::var(&request.hf_token_env)
        .ok()
        .filter(|s| !s.trim().is_empty());
    require_gated_token_if_needed(
        &targets,
        &manifest_by_id,
        &request.hf_token_env,
        token.as_deref(),
    )?;

    let mut lock_models = Vec::new();
    let mut installed = Vec::new();
    let mut skipped = Vec::new();

    for model in targets {
        let revision = match model {
            KnownModel::Rmbg14 => request.revision_rmbg14.as_str(),
            KnownModel::Rmbg20 => request.revision_rmbg20.as_str(),
        };
        let model_id = model.model_id().to_string();
        let rev_dir = model_revision_dir(&paths, model, revision)?;

        let lock_model = if rev_dir.exists() {
            if has_onnx_file(&rev_dir)? {
                skipped.push(model_id.clone());
                lock_from_existing_dir(&model_id, revision, &rev_dir)?
            } else {
                remove_revision_dir(&paths.models_dir, &rev_dir)?;
                let downloaded = download_model_to_revision(
                    &paths.cache_downloads_dir,
                    &model_id,
                    revision,
                    token.as_deref(),
                    &rev_dir,
                    request.onnx_variant,
                )?;
                installed.push(model_id.clone());
                downloaded
            }
        } else {
            let downloaded = download_model_to_revision(
                &paths.cache_downloads_dir,
                &model_id,
                revision,
                token.as_deref(),
                &rev_dir,
                request.onnx_variant,
            )?;
            installed.push(model_id.clone());
            downloaded
        };
        lock_models.push(lock_model);
    }

    let mut lockfile_written = false;
    if !request.verify_only {
        validate_lock_models(&paths, &lock_models)?;
        let existing = read_lockfile(&paths).ok();
        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());
        let lock = merge_lock_models(existing, lock_models, generated_at);
        write_lockfile(&paths, &lock)?;
        lockfile_written = true;
    }

    Ok(InstallReport {
        model_dir: paths.root,
        installed,
        skipped,
        lockfile_written,
    })
}

pub fn verify_models(model_dir: Option<PathBuf>) -> Result<ModelLock> {
    let paths = resolve_model_paths(model_dir.as_deref())?;
    let lock = read_lockfile(&paths)?;
    for model in &lock.models {
        let model_kind = unbg_model_registry::KnownModel::from_model_id(&model.model_id)
            .ok_or_else(|| anyhow!("unknown model id in lockfile: {}", model.model_id))?;
        let revision_dir = model_revision_dir(&paths, model_kind, &model.revision)?;
        for file in &model.files {
            let file_path = safe_join(&revision_dir, &file.path, "model lockfile entry")?;
            if !file_path.exists() {
                return Err(anyhow!(
                    "missing file for {}@{}: {}",
                    model.model_id,
                    model.revision,
                    file.path
                ));
            }
            let metadata = fs::metadata(&file_path)?;
            if metadata.len() != file.size {
                return Err(anyhow!(
                    "size mismatch for {}@{} {}: expected {}, got {}",
                    model.model_id,
                    model.revision,
                    file.path,
                    file.size,
                    metadata.len()
                ));
            }
            let digest = sha256_file(&file_path)?;
            if digest != file.sha256 {
                return Err(anyhow!(
                    "checksum mismatch for {}@{} {}",
                    model.model_id,
                    model.revision,
                    file.path
                ));
            }
        }
    }
    Ok(lock)
}

fn has_onnx_file(revision_dir: &Path) -> Result<bool> {
    for entry in WalkDir::new(revision_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        if entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("onnx"))
            .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn require_gated_token_if_needed(
    targets: &[KnownModel],
    manifest_by_id: &HashMap<String, unbg_model_registry::ModelManifest>,
    token_env: &str,
    token: Option<&str>,
) -> Result<()> {
    for model in targets {
        let model_id = model.model_id();
        let gated = manifest_by_id
            .get(model_id)
            .map(|m| m.gated)
            .ok_or_else(|| anyhow!("model not found in manifest: {}", model_id))?;
        if gated && token.is_none() {
            return Err(anyhow!(
                "missing {} for gated model {}",
                token_env,
                model_id
            ));
        }
    }
    Ok(())
}

fn download_model_to_revision(
    cache_downloads_dir: &Path,
    model_id: &str,
    revision: &str,
    token: Option<&str>,
    final_revision_dir: &Path,
    onnx_variant: OnnxVariant,
) -> Result<LockModel> {
    let client = hf_client(token)?;
    let files = list_model_files(&client, model_id, revision, onnx_variant)?;
    if files.is_empty() {
        return Err(anyhow!("no files listed for {}@{}", model_id, revision));
    }

    fs::create_dir_all(
        final_revision_dir
            .parent()
            .ok_or_else(|| anyhow!("invalid final revision directory"))?,
    )?;

    let tempdir = Builder::new()
        .prefix("unbg-download-")
        .tempdir_in(cache_downloads_dir)?;
    let temp_path = tempdir.path().to_path_buf();

    let mut lock_entries = Vec::with_capacity(files.len());
    for relative_path in files {
        let local_path = safe_join(&temp_path, &relative_path, "downloaded model file")?;
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let (size, sha256) =
            download_file(&client, model_id, revision, &relative_path, &local_path)?;
        lock_entries.push(LockFileEntry {
            path: relative_path,
            size,
            sha256,
        });
    }

    let kept = tempdir.keep();
    fs::rename(&kept, final_revision_dir).inspect_err(|_| {
        let _ = fs::remove_dir_all(&kept);
    })?;

    Ok(LockModel {
        model_id: model_id.to_string(),
        revision: revision.to_string(),
        source: "huggingface".to_string(),
        files: lock_entries,
    })
}

fn lock_from_existing_dir(
    model_id: &str,
    revision: &str,
    revision_dir: &Path,
) -> Result<LockModel> {
    let mut files = Vec::new();
    for entry in WalkDir::new(revision_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let full_path = entry.path();
        let rel = full_path
            .strip_prefix(revision_dir)
            .context("failed to strip revision dir prefix")?
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = fs::metadata(full_path)?;
        files.push(LockFileEntry {
            path: rel,
            size: metadata.len(),
            sha256: sha256_file(full_path)?,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(LockModel {
        model_id: model_id.to_string(),
        revision: revision.to_string(),
        source: "huggingface".to_string(),
        files,
    })
}

fn hf_client(token: Option<&str>) -> Result<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("unbg-installer/0.1"));
    if let Some(token) = token {
        let value = format!("Bearer {}", token);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&value).context("invalid token for authorization header")?,
        );
    }
    Ok(Client::builder()
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(30 * 60))
        .build()?)
}

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

fn list_model_files(
    client: &Client,
    model_id: &str,
    revision: &str,
    onnx_variant: OnnxVariant,
) -> Result<Vec<String>> {
    let url = huggingface_url(model_id, "tree", revision, None)?;
    let response = client.get(url).send()?;
    if !response.status().is_success() {
        return Err(anyhow!(
            "failed to list files for {}@{}: {}",
            model_id,
            revision,
            response.status()
        ));
    }
    if response.content_length().unwrap_or(0) > MAX_TREE_RESPONSE_BYTES {
        return Err(anyhow!("model file listing exceeded the response limit"));
    }
    let mut body = Vec::new();
    response
        .take(MAX_TREE_RESPONSE_BYTES + 1)
        .read_to_end(&mut body)?;
    if body.len() as u64 > MAX_TREE_RESPONSE_BYTES {
        return Err(anyhow!("model file listing exceeded the response limit"));
    }
    let entries: Vec<HfTreeEntry> = serde_json::from_slice(&body)?;
    let all_files: Vec<String> = entries
        .into_iter()
        .filter(|entry| entry.kind == "file")
        .map(|entry| entry.path)
        .collect();
    Ok(filter_model_files_for_variant(&all_files, onnx_variant))
}

fn filter_model_files_for_variant(all_files: &[String], onnx_variant: OnnxVariant) -> Vec<String> {
    let mut onnx_files: Vec<String> = all_files
        .iter()
        .filter(|f| f.starts_with("onnx/") && f.ends_with(".onnx"))
        .cloned()
        .collect();
    onnx_files.sort();

    let pick = |needle: &str| {
        onnx_files
            .iter()
            .find(|f| f.to_ascii_lowercase().contains(needle))
            .cloned()
    };

    let preferred = match onnx_variant {
        OnnxVariant::Fp16 => pick("fp16")
            .or_else(|| pick("model.onnx"))
            .or_else(|| onnx_files.first().cloned()),
        OnnxVariant::Fp32 => pick("model.onnx")
            .or_else(|| pick("fp16"))
            .or_else(|| onnx_files.first().cloned()),
        OnnxVariant::Quantized => pick("quantized")
            .or_else(|| pick("q8"))
            .or_else(|| onnx_files.first().cloned()),
        OnnxVariant::Auto => pick("fp16")
            .or_else(|| pick("model.onnx"))
            .or_else(|| onnx_files.first().cloned()),
    };

    let mut out = Vec::new();
    if let Some(file) = preferred {
        out.push(file);
    }
    for meta in ["config.json", "preprocessor_config.json"] {
        if all_files.iter().any(|f| f == meta) {
            out.push(meta.to_string());
        }
    }
    out
}

fn download_file(
    client: &Client,
    model_id: &str,
    revision: &str,
    file_path: &str,
    destination: &Path,
) -> Result<(u64, String)> {
    let url = huggingface_url(model_id, "resolve", revision, Some(file_path))?;
    let partial_path = destination.with_extension("part");
    let resume_from = fs::metadata(&partial_path).map(|m| m.len()).unwrap_or(0);
    if resume_from > MAX_MODEL_FILE_BYTES {
        return Err(anyhow!(
            "partial download for {file_path} exceeds the {} byte limit",
            MAX_MODEL_FILE_BYTES
        ));
    }
    let mut response = if resume_from > 0 {
        client
            .get(url.clone())
            .header(RANGE, format!("bytes={}-", resume_from))
            .send()?
    } else {
        client.get(url.clone()).send()?
    };
    if resume_from > 0 && response.status().as_u16() == 200 {
        let _ = fs::remove_file(&partial_path);
        response = client.get(url).send()?;
    }
    if !response.status().is_success() {
        return Err(anyhow!(
            "failed downloading {} for {}@{}: {}",
            file_path,
            model_id,
            revision,
            response.status()
        ));
    }
    if response
        .content_length()
        .and_then(|remaining| resume_from.checked_add(remaining))
        .is_some_and(|total| total > MAX_MODEL_FILE_BYTES)
    {
        return Err(anyhow!(
            "download for {file_path} exceeds the {} byte limit",
            MAX_MODEL_FILE_BYTES
        ));
    }

    let mut file = if resume_from > 0 {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial_path)?
    } else {
        File::create(&partial_path)?
    };
    let mut hasher = Sha256::new();
    if resume_from > 0 {
        let mut existing = File::open(&partial_path)?;
        let mut existing_buf = [0u8; 16 * 1024];
        loop {
            let read = existing.read(&mut existing_buf)?;
            if read == 0 {
                break;
            }
            hasher.update(&existing_buf[..read]);
        }
    }
    let mut buf = [0u8; 16 * 1024];
    let mut total_size = resume_from;
    loop {
        let read = response.read(&mut buf)?;
        if read == 0 {
            break;
        }
        total_size = total_size
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("download size overflow for {file_path}"))?;
        if total_size > MAX_MODEL_FILE_BYTES {
            drop(file);
            let _ = fs::remove_file(&partial_path);
            return Err(anyhow!(
                "download for {file_path} exceeds the {} byte limit",
                MAX_MODEL_FILE_BYTES
            ));
        }
        file.write_all(&buf[..read])?;
        hasher.update(&buf[..read]);
    }
    file.flush()?;
    fs::rename(&partial_path, destination)?;
    let total_size = fs::metadata(destination)?.len();
    let digest = hex::encode(hasher.finalize());
    Ok((total_size, digest))
}

fn huggingface_url(
    model_id: &str,
    operation: &str,
    revision: &str,
    file_path: Option<&str>,
) -> Result<Url> {
    let mut url = Url::parse("https://huggingface.co")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("Hugging Face base URL cannot contain path segments"))?;
        if operation == "tree" {
            segments.push("api").push("models");
        }
        for segment in model_id.split('/') {
            segments.push(segment);
        }
        segments.push(operation);
        for segment in revision.split('/') {
            segments.push(segment);
        }
        if let Some(file_path) = file_path {
            for segment in file_path.split('/') {
                segments.push(segment);
            }
        }
    }
    if operation == "tree" {
        url.query_pairs_mut().append_pair("recursive", "1");
    }
    Ok(url)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn validate_lock_models(
    paths: &unbg_model_registry::ModelPaths,
    models: &[LockModel],
) -> Result<()> {
    for model in models {
        if model.files.is_empty() {
            return Err(anyhow!("model {} has no tracked files", model.model_id));
        }
        let kind = KnownModel::from_model_id(&model.model_id)
            .ok_or_else(|| anyhow!("unknown model id in lock entries: {}", model.model_id))?;
        let revision_dir = model_revision_dir(paths, kind, &model.revision)?;
        if !has_onnx_file(&revision_dir)? {
            return Err(anyhow!(
                "revision {} for {} has no onnx file",
                model.revision,
                model.model_id
            ));
        }
        for entry in &model.files {
            let full = safe_join(&revision_dir, &entry.path, "model lockfile entry")?;
            if !full.exists() {
                return Err(anyhow!(
                    "missing file before lock write: {}",
                    full.display()
                ));
            }
        }
    }
    Ok(())
}

fn remove_revision_dir(models_dir: &Path, revision_dir: &Path) -> Result<()> {
    let canonical_models =
        fs::canonicalize(models_dir).context("failed to resolve models directory")?;
    let canonical_revision =
        fs::canonicalize(revision_dir).context("failed to resolve model revision directory")?;
    if canonical_revision == canonical_models || !canonical_revision.starts_with(&canonical_models)
    {
        return Err(anyhow!(
            "refusing to remove model revision outside {}: {}",
            canonical_models.display(),
            canonical_revision.display()
        ));
    }
    fs::remove_dir_all(revision_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_removal_is_confined_to_models_directory() {
        let temp = tempfile::tempdir().expect("temp directory should be created");
        let models = temp.path().join("models");
        let inside = models.join("briaai__RMBG-1.4").join("main");
        let outside = temp.path().join("outside");
        fs::create_dir_all(&inside).expect("inside revision should be created");
        fs::create_dir_all(&outside).expect("outside directory should be created");

        assert!(remove_revision_dir(&models, &outside).is_err());
        assert!(outside.exists(), "outside directory must not be removed");

        remove_revision_dir(&models, &inside).expect("inside revision should be removed");
        assert!(!inside.exists());
    }

    #[test]
    fn huggingface_urls_encode_untrusted_path_characters() {
        let tree = huggingface_url("briaai/RMBG-1.4", "tree", "refs/pr/1", None)
            .expect("tree URL should build");
        assert_eq!(tree.host_str(), Some("huggingface.co"));
        assert_eq!(tree.query(), Some("recursive=1"));

        let download = huggingface_url(
            "briaai/RMBG-1.4",
            "resolve",
            "main",
            Some("onnx/model name.onnx"),
        )
        .expect("download URL should build");
        assert!(download.as_str().contains("model%20name.onnx"));
    }
}
