use crate::{
    event::{ParseLineError, TuneEventEnvelope},
    state::{AppState, LogLine},
};

#[derive(Debug, Clone)]
pub enum AppMsg {
    StdinEvent(TuneEventEnvelope),
    StdinParseError { line: String, kind: ParseLineError },
    StdinClosed,
}

pub fn reduce(state: &mut AppState, msg: AppMsg) {
    match msg {
        AppMsg::StdinEvent(env) => {
            if state.run_id.is_none() {
                state.run_id = Some(env.run_id.clone());
            }
            if state.schema_version.is_none() {
                state.schema_version = Some(env.schema_version.clone());
            }

            state.all_events.push(env.clone());

            let event_name = env.event.as_str();
            if is_run_phase_event(&env) {
                state.progress_events.push(env.clone());
            }
            if event_name == "profile.snapshot" {
                state.profiling_events.push(env.clone());
            }
            if event_name.starts_with("recommendation.")
                || event_name.starts_with("action.")
                || event_name.contains("recommend")
            {
                state.action_events.push(env.clone());
            }
            if is_candidate_or_evidence_event(&env) {
                state.evidence_events.push(env.clone());
            }
            if event_name.starts_with("warning.")
                || event_name == "error"
                || env.event_type == "log"
                || env.event_type == "error"
            {
                let msg = render_log_line(&env);
                state.log_lines.push(msg);
            }

            clamp_selection(state);
        }
        AppMsg::StdinParseError { line, kind } => {
            state.parse_error_count += 1;
            state.log_lines.push(LogLine {
                seq: 0,
                message: format!("parse error ({kind:?}): {line}"),
            });
            clamp_selection(state);
        }
        AppMsg::StdinClosed => {}
    }
}

fn clamp_selection(state: &mut AppState) {
    state.selected_progress = clamp_idx(state.selected_progress, state.progress_events.len());
    state.selected_profiling = clamp_idx(state.selected_profiling, state.profiling_events.len());
    state.selected_actions = clamp_idx(state.selected_actions, state.action_events.len());
    state.selected_evidence = clamp_idx(state.selected_evidence, state.evidence_events.len());
    state.selected_logs = clamp_idx(state.selected_logs, state.log_lines.len());
}

fn clamp_idx(idx: usize, len: usize) -> usize {
    if len == 0 {
        0
    } else if idx >= len {
        len - 1
    } else {
        idx
    }
}

fn render_log_line(env: &TuneEventEnvelope) -> LogLine {
    let message = env
        .payload
        .get("message")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| env.event.clone());

    LogLine {
        seq: env.seq,
        message,
    }
}

fn is_run_phase_event(env: &TuneEventEnvelope) -> bool {
    let event_name = env.event.as_str();
    if event_name.starts_with("run.") {
        return true;
    }

    if !event_name.starts_with("phase.") {
        return false;
    }

    let phase_name = env
        .payload
        .get("phase")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    phase_name.contains("baseline")
        || phase_name.contains("iteration")
        || phase_name.contains("run")
        || phase_name.contains("attach")
}

fn is_candidate_or_evidence_event(env: &TuneEventEnvelope) -> bool {
    let event_name = env.event.as_str();
    if event_name == "candidate.evaluated" {
        return env.payload.get("wall_time_s").and_then(|v| v.as_f64()).is_some();
    }

    if event_name == "phase.finished" {
        let phase_name = env
            .payload
            .get("phase")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !phase_name.contains("baseline") {
            return false;
        }
        return env.payload.get("wall_time_s").and_then(|v| v.as_f64()).is_some();
    }

    false
}
