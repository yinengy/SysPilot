use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneEventEnvelope {
    pub schema_version: String,
    pub run_id: String,
    pub seq: u64,
    pub ts: String,
    #[serde(default = "default_event_type")]
    pub event_type: String,
    pub event: String,
    #[serde(default = "default_payload")]
    pub payload: serde_json::Value,
}

fn default_event_type() -> String {
    "event".to_string()
}

fn default_payload() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseLineError {
    Empty,
    InvalidJson,
    WrongShape,
    InvalidEnvelope,
}

pub fn parse_event_line(line: &str) -> Result<Option<TuneEventEnvelope>, ParseLineError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|_| ParseLineError::InvalidJson)?;
    let mut env: TuneEventEnvelope =
        serde_json::from_value(value).map_err(|_| ParseLineError::WrongShape)?;

    if env.schema_version.trim().is_empty()
        || env.run_id.trim().is_empty()
        || env.seq == 0
        || env.ts.trim().is_empty()
        || env.event_type.trim().is_empty()
        || env.event.trim().is_empty()
    {
        return Err(ParseLineError::InvalidEnvelope);
    }

    if !env.payload.is_object() {
        env.payload = default_payload();
    }
    Ok(Some(env))
}
