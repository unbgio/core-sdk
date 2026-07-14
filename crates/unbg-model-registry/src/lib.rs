use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LOCKFILE_NAME: &str = "unbg-model-lock.json";
pub const SCHEMA_VERSION: u32 = 1;
const MAX_LOCKFILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_LOCK_MODELS: usize = 16;
const MAX_MODEL_FILES: usize = 4_096;
const MAX_REVISION_LENGTH: usize = 256;
const MAX_RELATIVE_PATH_LENGTH: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPreprocessing {
    pub input_width: u32,
    pub input_height: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KnownModel {
    Rmbg14,
    Rmbg20,
}

impl KnownModel {
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Rmbg14 => "briaai/RMBG-1.4",
            Self::Rmbg20 => "briaai/RMBG-2.0",
        }
    }

    pub fn cache_key(self) -> &'static str {
        match self {
            Self::Rmbg14 => "briaai__RMBG-1.4",
            Self::Rmbg20 => "briaai__RMBG-2.0",
        }
    }

    pub fn all() -> [KnownModel; 2] {
        [KnownModel::Rmbg14, KnownModel::Rmbg20]
    }

    pub fn from_model_id(model_id: &str) -> Option<Self> {
        match model_id {
            "briaai/RMBG-1.4" => Some(Self::Rmbg14),
            "briaai/RMBG-2.0" => Some(Self::Rmbg20),
            _ => None,
        }
    }

    pub fn preprocessing(self) -> ModelPreprocessing {
        match self {
            Self::Rmbg14 => ModelPreprocessing {
                input_width: 1024,
                input_height: 1024,
                mean: [0.5, 0.5, 0.5],
                std: [1.0, 1.0, 1.0],
            },
            Self::Rmbg20 => ModelPreprocessing {
                input_width: 1024,
                input_height: 1024,
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            },
        }
    }
}

impl Display for KnownModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.model_id())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LockFileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LockModel {
    #[serde(alias = "model_id")]
    pub model_id: String,
    pub revision: String,
    pub source: String,
    pub files: Vec<LockFileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelLock {
    #[serde(alias = "schema_version")]
    pub schema_version: u32,
    #[serde(alias = "generated_at")]
    pub generated_at: String,
    pub models: Vec<LockModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManifest {
    pub model_id: String,
    pub default_revision: String,
    pub gated: bool,
}

pub fn built_in_manifest() -> Vec<ModelManifest> {
    vec![
        ModelManifest {
            model_id: "briaai/RMBG-1.4".to_string(),
            default_revision: "main".to_string(),
            gated: false,
        },
        ModelManifest {
            model_id: "briaai/RMBG-2.0".to_string(),
            default_revision: "main".to_string(),
            gated: true,
        },
    ]
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("could not determine a default model directory")]
    NoDefaultModelDir,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid {kind} path: {value}")]
    InvalidRelativePath { kind: &'static str, value: String },
    #[error("model lockfile is too large: {actual} bytes (maximum {maximum})")]
    LockfileTooLarge { actual: u64, maximum: u64 },
    #[error("invalid model lockfile: {0}")]
    InvalidLockfile(String),
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub root: PathBuf,
    pub manifests_dir: PathBuf,
    pub models_dir: PathBuf,
    pub cache_downloads_dir: PathBuf,
}

pub fn default_model_dir() -> Result<PathBuf, RegistryError> {
    let home = dirs::home_dir().ok_or(RegistryError::NoDefaultModelDir)?;
    Ok(home.join(".unbg").join("models"))
}

pub fn resolve_model_paths(model_dir: Option<&Path>) -> Result<ModelPaths, RegistryError> {
    let root = if let Some(dir) = model_dir {
        dir.to_path_buf()
    } else {
        default_model_dir()?
    };

    Ok(ModelPaths {
        manifests_dir: root.join("manifests"),
        models_dir: root.join("models"),
        cache_downloads_dir: root.join("cache").join("downloads"),
        root,
    })
}

pub fn ensure_layout(paths: &ModelPaths) -> Result<(), RegistryError> {
    fs::create_dir_all(&paths.manifests_dir)?;
    fs::create_dir_all(&paths.models_dir)?;
    fs::create_dir_all(&paths.cache_downloads_dir)?;
    Ok(())
}

pub fn validate_relative_path(value: &str, kind: &'static str) -> Result<(), RegistryError> {
    let invalid = || RegistryError::InvalidRelativePath {
        kind,
        value: value.to_string(),
    };

    if value.is_empty()
        || value.len() > MAX_RELATIVE_PATH_LENGTH
        || value.contains('\0')
        || value.contains('\\')
        || value.contains(':')
        || value
            .chars()
            .any(|character| matches!(character, '<' | '>' | '"' | '|' | '?' | '*'))
        || value.chars().any(char::is_control)
        || value.starts_with('/')
        || value.ends_with('/')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(invalid());
    }

    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid());
    }

    Ok(())
}

pub fn safe_join(
    base: &Path,
    relative: &str,
    kind: &'static str,
) -> Result<PathBuf, RegistryError> {
    validate_relative_path(relative, kind)?;
    Ok(base.join(relative))
}

pub fn model_revision_dir(
    paths: &ModelPaths,
    model: KnownModel,
    revision: &str,
) -> Result<PathBuf, RegistryError> {
    if revision.len() > MAX_REVISION_LENGTH {
        return Err(RegistryError::InvalidRelativePath {
            kind: "model revision",
            value: revision.to_string(),
        });
    }
    if !revision
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
    {
        return Err(RegistryError::InvalidRelativePath {
            kind: "model revision",
            value: revision.to_string(),
        });
    }
    let model_dir = paths.models_dir.join(model.cache_key());
    safe_join(&model_dir, revision, "model revision")
}

pub fn lockfile_path(paths: &ModelPaths) -> PathBuf {
    paths.manifests_dir.join(LOCKFILE_NAME)
}

pub fn write_lockfile(paths: &ModelPaths, lock: &ModelLock) -> Result<(), RegistryError> {
    ensure_layout(paths)?;
    validate_lockfile(lock)?;
    let data = serde_json::to_vec_pretty(lock)?;
    if data.len() as u64 > MAX_LOCKFILE_BYTES {
        return Err(RegistryError::LockfileTooLarge {
            actual: data.len() as u64,
            maximum: MAX_LOCKFILE_BYTES,
        });
    }
    fs::write(lockfile_path(paths), data)?;
    Ok(())
}

pub fn read_lockfile(paths: &ModelPaths) -> Result<ModelLock, RegistryError> {
    let path = lockfile_path(paths);
    let size = fs::metadata(&path)?.len();
    if size > MAX_LOCKFILE_BYTES {
        return Err(RegistryError::LockfileTooLarge {
            actual: size,
            maximum: MAX_LOCKFILE_BYTES,
        });
    }
    let data = fs::read(path)?;
    if data.len() as u64 > MAX_LOCKFILE_BYTES {
        return Err(RegistryError::LockfileTooLarge {
            actual: data.len() as u64,
            maximum: MAX_LOCKFILE_BYTES,
        });
    }
    let lock: ModelLock = serde_json::from_slice(&data)?;
    validate_lockfile(&lock)?;
    Ok(lock)
}

pub fn validate_lockfile(lock: &ModelLock) -> Result<(), RegistryError> {
    if lock.schema_version != SCHEMA_VERSION {
        return Err(RegistryError::InvalidLockfile(format!(
            "unsupported schema version {} (expected {})",
            lock.schema_version, SCHEMA_VERSION
        )));
    }
    if lock.models.len() > MAX_LOCK_MODELS {
        return Err(RegistryError::InvalidLockfile(format!(
            "contains {} models (maximum {})",
            lock.models.len(),
            MAX_LOCK_MODELS
        )));
    }

    let mut model_ids = std::collections::HashSet::new();
    for model in &lock.models {
        if KnownModel::from_model_id(&model.model_id).is_none() {
            return Err(RegistryError::InvalidLockfile(format!(
                "unknown model id {}",
                model.model_id
            )));
        }
        if !model_ids.insert(model.model_id.as_str()) {
            return Err(RegistryError::InvalidLockfile(format!(
                "duplicate model id {}",
                model.model_id
            )));
        }
        if model.revision.len() > MAX_REVISION_LENGTH {
            return Err(RegistryError::InvalidLockfile(format!(
                "revision is too long for {}",
                model.model_id
            )));
        }
        validate_relative_path(&model.revision, "model revision")?;
        if model.source.is_empty()
            || model.source.len() > 256
            || model.source.chars().any(char::is_control)
        {
            return Err(RegistryError::InvalidLockfile(format!(
                "invalid source for {}",
                model.model_id
            )));
        }
        if model.files.is_empty() || model.files.len() > MAX_MODEL_FILES {
            return Err(RegistryError::InvalidLockfile(format!(
                "{} has {} files (expected 1..={})",
                model.model_id,
                model.files.len(),
                MAX_MODEL_FILES
            )));
        }

        let mut file_paths = std::collections::HashSet::new();
        for file in &model.files {
            validate_relative_path(&file.path, "model lockfile entry")?;
            if !file_paths.insert(file.path.as_str()) {
                return Err(RegistryError::InvalidLockfile(format!(
                    "duplicate file path {} for {}",
                    file.path, model.model_id
                )));
            }
            if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(RegistryError::InvalidLockfile(format!(
                    "invalid sha256 for {} in {}",
                    file.path, model.model_id
                )));
            }
        }
    }
    Ok(())
}

pub fn merge_lock_models(
    existing: Option<ModelLock>,
    updates: Vec<LockModel>,
    generated_at: String,
) -> ModelLock {
    let mut by_id = std::collections::BTreeMap::new();
    if let Some(lock) = existing {
        for model in lock.models {
            by_id.insert(model.model_id.clone(), model);
        }
    }
    for model in updates {
        by_id.insert(model.model_id.clone(), model);
    }
    ModelLock {
        schema_version: SCHEMA_VERSION,
        generated_at,
        models: by_id.into_values().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_replaces_existing_model_entry() {
        let existing = ModelLock {
            schema_version: SCHEMA_VERSION,
            generated_at: "1".to_string(),
            models: vec![LockModel {
                model_id: "briaai/RMBG-1.4".to_string(),
                revision: "old".to_string(),
                source: "huggingface".to_string(),
                files: vec![],
            }],
        };

        let merged = merge_lock_models(
            Some(existing),
            vec![LockModel {
                model_id: "briaai/RMBG-1.4".to_string(),
                revision: "new".to_string(),
                source: "huggingface".to_string(),
                files: vec![],
            }],
            "2".to_string(),
        );

        assert_eq!(merged.models.len(), 1);
        assert_eq!(merged.models[0].revision, "new");
    }

    #[test]
    fn preprocessing_metadata_is_model_specific() {
        let rmbg14 = KnownModel::Rmbg14.preprocessing();
        let rmbg20 = KnownModel::Rmbg20.preprocessing();

        assert_eq!(rmbg14.mean, [0.5, 0.5, 0.5]);
        assert_eq!(rmbg14.std, [1.0, 1.0, 1.0]);
        assert_eq!(rmbg20.mean, [0.485, 0.456, 0.406]);
        assert_eq!(rmbg20.std, [0.229, 0.224, 0.225]);
        assert_eq!((rmbg14.input_width, rmbg14.input_height), (1024, 1024));
        assert_eq!((rmbg20.input_width, rmbg20.input_height), (1024, 1024));
    }

    #[test]
    fn model_revision_path_accepts_nested_safe_revisions() {
        let paths =
            resolve_model_paths(Some(Path::new("models-root"))).expect("paths should resolve");
        let revision = model_revision_dir(&paths, KnownModel::Rmbg14, "refs/pr/123")
            .expect("safe revision should resolve");

        assert_eq!(
            revision,
            paths
                .models_dir
                .join(KnownModel::Rmbg14.cache_key())
                .join("refs/pr/123")
        );
    }

    #[test]
    fn model_revision_path_rejects_traversal_and_absolute_paths() {
        let paths =
            resolve_model_paths(Some(Path::new("models-root"))).expect("paths should resolve");
        for revision in [
            "../outside",
            "main/../../outside",
            "/absolute",
            r"C:\outside",
            r"main\..\outside",
            "main//nested",
            ".",
            "",
        ] {
            assert!(
                model_revision_dir(&paths, KnownModel::Rmbg14, revision).is_err(),
                "revision should be rejected: {revision:?}"
            );
        }
    }

    #[test]
    fn safe_join_rejects_untrusted_lockfile_paths() {
        let base = Path::new("model-revision");
        assert_eq!(
            safe_join(base, "onnx/model.onnx", "model file").expect("safe path should resolve"),
            base.join("onnx/model.onnx")
        );

        for path in [
            "../secret",
            "/secret",
            r"C:\secret",
            r"onnx\..\secret",
            "onnx/../../secret",
        ] {
            assert!(
                safe_join(base, path, "model file").is_err(),
                "path should be rejected: {path:?}"
            );
        }
    }

    #[test]
    fn lockfile_validation_rejects_untrusted_and_duplicate_entries() {
        let valid_file = LockFileEntry {
            path: "onnx/model.onnx".to_string(),
            size: 42,
            sha256: "a".repeat(64),
        };
        let mut lock = ModelLock {
            schema_version: SCHEMA_VERSION,
            generated_at: "test".to_string(),
            models: vec![LockModel {
                model_id: KnownModel::Rmbg14.model_id().to_string(),
                revision: "main".to_string(),
                source: "huggingface".to_string(),
                files: vec![valid_file.clone()],
            }],
        };
        assert!(validate_lockfile(&lock).is_ok());

        lock.models[0].files.push(valid_file);
        assert!(validate_lockfile(&lock).is_err());
        lock.models[0].files[1].path = "../outside.onnx".to_string();
        assert!(validate_lockfile(&lock).is_err());
    }
}
