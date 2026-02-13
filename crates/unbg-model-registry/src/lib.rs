use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LOCKFILE_NAME: &str = "unbg-model-lock.json";
pub const SCHEMA_VERSION: u32 = 1;

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

pub fn model_revision_dir(paths: &ModelPaths, model: KnownModel, revision: &str) -> PathBuf {
    paths.models_dir.join(model.cache_key()).join(revision)
}

pub fn lockfile_path(paths: &ModelPaths) -> PathBuf {
    paths.manifests_dir.join(LOCKFILE_NAME)
}

pub fn write_lockfile(paths: &ModelPaths, lock: &ModelLock) -> Result<(), RegistryError> {
    ensure_layout(paths)?;
    let data = serde_json::to_vec_pretty(lock)?;
    fs::write(lockfile_path(paths), data)?;
    Ok(())
}

pub fn read_lockfile(paths: &ModelPaths) -> Result<ModelLock, RegistryError> {
    let data = fs::read(lockfile_path(paths))?;
    Ok(serde_json::from_slice(&data)?)
}

pub fn merge_lock_models(existing: Option<ModelLock>, updates: Vec<LockModel>, generated_at: String) -> ModelLock {
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
}
