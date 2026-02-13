use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::Serialize;
use unbg_core::{TelemetryEvent, TelemetrySink};

pub fn sink_from_env() -> Option<Box<dyn TelemetrySink>> {
    let mode = std::env::var("UNBG_TELEMETRY_SINK").ok()?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "stdout" => Some(Box::new(StdoutSink)),
        "file" => {
            let path = std::env::var("UNBG_TELEMETRY_FILE").ok().filter(|v| !v.trim().is_empty())?;
            Some(Box::new(FileSink::new(PathBuf::from(path))))
        }
        "http" => {
            let endpoint = std::env::var("UNBG_TELEMETRY_ENDPOINT")
                .ok()
                .filter(|v| !v.trim().is_empty())?;
            Some(Box::new(HttpSink::new(endpoint)))
        }
        _ => None,
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TelemetryEnvelope {
    event_type: String,
    model: String,
    platform: String,
    duration_ms: Option<u64>,
    detail: Option<String>,
}

impl From<&TelemetryEvent> for TelemetryEnvelope {
    fn from(event: &TelemetryEvent) -> Self {
        Self {
            event_type: format!("{:?}", event.event_type),
            model: format!("{:?}", event.model),
            platform: format!("{:?}", event.platform),
            duration_ms: event.duration_ms,
            detail: event.detail.clone(),
        }
    }
}

pub struct StdoutSink;

impl TelemetrySink for StdoutSink {
    fn emit(&self, event: TelemetryEvent) {
        if let Ok(line) = serde_json::to_string(&TelemetryEnvelope::from(&event)) {
            println!("{}", line);
        }
    }
}

pub struct FileSink {
    path: PathBuf,
}

impl FileSink {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn write_line(&self, line: &str) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).context("creating telemetry log parent directory")?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .context("opening telemetry file")?;
        writeln!(file, "{}", line).context("writing telemetry line")?;
        Ok(())
    }
}

impl TelemetrySink for FileSink {
    fn emit(&self, event: TelemetryEvent) {
        if let Ok(line) = serde_json::to_string(&TelemetryEnvelope::from(&event)) {
            let _ = self.write_line(&line);
        }
    }
}

pub struct HttpSink {
    endpoint: String,
    client: Client,
}

impl HttpSink {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: Client::new(),
        }
    }
}

impl TelemetrySink for HttpSink {
    fn emit(&self, event: TelemetryEvent) {
        let payload = TelemetryEnvelope::from(&event);
        let _ = self.client.post(&self.endpoint).json(&payload).send();
    }
}
