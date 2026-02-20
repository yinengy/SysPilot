use std::path::PathBuf;

use syspilot_tui::{
    event::{parse_event_line, ParseLineError},
    reducer::{reduce, AppMsg},
    state::{AppState, BoundedDeque},
};

fn read_fixture(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("tests").join("fixtures").join(name);
    std::fs::read_to_string(path).expect("fixture readable")
}

#[test]
fn parse_event_line_rejects_invalid_envelope() {
    let line = r#"{"schema_version":"v1","run_id":"r","seq":0,"ts":"t","event_type":"x","event":"run.started","payload":{}}"#;
    let err = parse_event_line(line).unwrap_err();
    assert_eq!(err, ParseLineError::InvalidEnvelope);
}

#[test]
fn reducer_ingests_fixture_and_updates_state() {
    let input = read_fixture("sample_tune_events.jsonl");
    let mut state = AppState::new();
    for raw in input.lines() {
        match parse_event_line(raw) {
            Ok(parsed) => {
                if let Some(env) = parsed {
                    reduce(&mut state, AppMsg::StdinEvent(env));
                }
            }
            Err(kind) => reduce(
                &mut state,
                AppMsg::StdinParseError {
                    line: raw.to_string(),
                    kind,
                },
            ),
        }
    }

    assert_eq!(state.run_id.as_deref(), Some("run-123"));
    assert_eq!(state.schema_version.as_deref(), Some("v1"));
    assert_eq!(state.parse_error_count, 0);
    assert_eq!(state.all_events.len(), 9);
    assert!(state.progress_events.len() >= 3);
    let progress_events: Vec<&str> = state.progress_events.iter().map(|event| event.event.as_str()).collect();
    assert_eq!(
        progress_events,
        vec![
            "run.started",
            "recommendation.ready",
            "candidate.evaluated",
        ]
    );
    assert!(state.action_events.len() >= 2);
    assert!(state.evidence_events.len() >= 2);
    assert!(state.log_lines.len() >= 1);
    assert_eq!(state.last_event().map(|e| e.event.as_str()), Some("run.finished"));
}

#[test]
fn malformed_fixture_does_not_panic_and_is_counted() {
    let input = read_fixture("sample_tune_events_malformed.jsonl");
    let mut state = AppState::new();
    for raw in input.lines() {
        match parse_event_line(raw) {
            Ok(parsed) => {
                if let Some(env) = parsed {
                    reduce(&mut state, AppMsg::StdinEvent(env));
                }
            }
            Err(kind) => reduce(
                &mut state,
                AppMsg::StdinParseError {
                    line: raw.to_string(),
                    kind,
                },
            ),
        }
    }
    assert!(state.parse_error_count >= 2);
    assert_eq!(state.last_event().map(|e| e.seq), Some(11));
}

#[test]
fn bounded_deque_enforces_capacity() {
    let mut q = BoundedDeque::new(3);
    q.push(1);
    q.push(2);
    q.push(3);
    q.push(4);
    assert_eq!(q.len(), 3);
    let got: Vec<_> = q.iter().copied().collect();
    assert_eq!(got, vec![2, 3, 4]);
}

#[test]
fn reducer_classifies_error_event_into_logs() {
    let mut state = AppState::new();
    let line = r#"{"schema_version":"v1","run_id":"run","seq":1,"ts":"2026-02-19T00:00:00Z","event_type":"error","event":"error","payload":{"message":"boom"}}"#;
    let env = parse_event_line(line).expect("parse ok").expect("event present");
    reduce(&mut state, AppMsg::StdinEvent(env));
    assert_eq!(state.log_lines.len(), 1);
}
