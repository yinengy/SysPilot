use std::{
    collections::{HashMap, HashSet},
    time::{SystemTime, UNIX_EPOCH},
};

use chrono::DateTime;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Bar, BarChart, BarGroup, Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::{
    event::TuneEventEnvelope,
    state::{AppState, Panel},
};

pub fn draw(f: &mut Frame, state: &AppState) {
    let area = f.area();
    let frame = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(frame[0]);

    let top_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(17),
            Constraint::Percentage(50),
        ])
        .split(rows[0]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Length(18), Constraint::Min(0)])
        .split(top_cols[0]);

    let mid_top = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(top_cols[1]);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    render_loop_state(f, left[0], state);
    render_run_phase(f, left[1], state);

    render_analysis_phase(f, mid_top[0], state);
    render_profiling_phase(f, mid_top[1], state);
    render_candidate_phase(f, bottom[0], state);

    render_decision_and_details(f, top_cols[2], state);
    render_logs(f, bottom[1], state);
    render_footer(f, frame[1]);
}

fn render_footer(f: &mut Frame, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let text =
        "F2: toggle mouse capture (disable for drag-select copy) | Ctrl+C: exit";
    let p = Paragraph::new(text).style(Style::default().fg(Color::Gray));
    f.render_widget(p, area);
}

fn panel_block(title: &str, focused: bool) -> Block {
    let style = if focused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(style)
}

fn render_loop_state(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Overview;
    let mut lines = Vec::new();

    let run_started = latest_by_name(state, "run.started");

    if let Some(started) = run_started {
        lines.push(Line::from(format!(
            "mode: {}",
            payload_str(started, "mode").unwrap_or("-")
        )));
        let cmd = payload_str(started, "command").unwrap_or("-");
        lines.push(Line::from(format!("cmd: {}", truncate(cmd, 52))));

        let loaded_manuals = payload_str_list(started, "loaded_manuals");
        if loaded_manuals.is_empty() {
            lines.push(Line::from("manuals: (none)"));
        } else {
            lines.push(Line::from(format!(
                "manuals: {}",
                truncate(&loaded_manuals.join(", "), 52)
            )));
        }
    } else {
        lines.push(Line::from("mode: -"));
        lines.push(Line::from("cmd: -"));
        lines.push(Line::from("manuals: -"));
    }

    let app_running = app_execution_running(state);
    let spinner = app_execution_spinner(app_running);
    let state_label = if app_running { "running" } else { "idle" };
    lines.push(Line::from(format!("state: {spinner} {state_label}")));

    let p = Paragraph::new(Text::from(lines))
        .block(panel_block("Loop State", focused))
        .wrap(Wrap { trim: true });
    f.render_widget(p, area);
}

fn render_run_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Progress;
    let block = panel_block("Run Phase", focused);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let content_width = inner.width.max(1) as usize;
    let header_messages = run_phase_header_messages(state);
    let header_rows = header_messages
        .iter()
        .map(|message| wrapped_line_count(message, content_width))
        .sum::<usize>()
        .min(inner.height as usize);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(header_rows as u16), Constraint::Min(0)])
        .split(inner);

    if header_rows > 0 {
        let header_lines: Vec<Line<'static>> = header_messages.into_iter().map(Line::from).collect();
        let header = Paragraph::new(Text::from(header_lines)).wrap(Wrap { trim: false });
        f.render_widget(header, layout[0]);
    }

    let entries_area = layout[1];
    if entries_area.height == 0 || entries_area.width == 0 {
        return;
    }

    let entry_messages = run_phase_entry_messages(state);
    let mut entry_lines = Vec::new();

    if entry_messages.is_empty() {
        entry_lines.push(Line::from("(no run events)"));
    } else {
        for (index, message) in entry_messages.iter().enumerate() {
            let selected = index == state.selected_progress;
            let style = if selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            entry_lines.push(Line::styled(message.clone(), style));
        }
    }

    let entry_width = entries_area.width.max(1) as usize;
    let max_scroll = max_wrapped_scroll_lines(&entry_messages, entry_width, entries_area.height as usize);
    let scroll_y = state
        .progress_scroll_lines
        .min(max_scroll)
        .min(u16::MAX as usize) as u16;

    let entries = Paragraph::new(Text::from(entry_lines))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(entries, entries_area);
}

pub(crate) fn run_phase_header_messages(_state: &AppState) -> Vec<String> {
    Vec::new()
}

pub(crate) fn run_phase_entry_messages(state: &AppState) -> Vec<String> {
    state
        .progress_events
        .iter()
        .map(|event| {
            let mut suffix = String::new();
            if let Some(status) = payload_str(event, "status") {
                suffix = format!(" [{status}]");
            }
            format!("{}{}", run_event_label(event), suffix)
        })
        .collect()
}

fn render_profiling_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Metrics;
    let mut lines = Vec::new();

    if state.profiling_events.is_empty() {
        lines.push(Line::from("(no profiling snapshot yet)"));
    } else {
        let max_rows = area.height.saturating_sub(2) as usize;
        let len = state.profiling_events.len();
        let start = clamped_index_window_start(state.profiling_scroll_start, len, max_rows);
        let end = (start + max_rows).min(len);

        for (idx, snapshot) in state
            .profiling_events
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            let selected = idx == state.selected_profiling;
            let style = if selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };

            let round = payload_u64(snapshot, "iteration").unwrap_or(0);

            lines.push(Line::styled(
                format!("round {round} profiling"),
                style,
            ));
        }
    }

    let p = Paragraph::new(Text::from(lines))
        .block(panel_block("Profiling Phase", focused));
    f.render_widget(p, area);
}

fn render_analysis_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Actions;
    let mut lines = Vec::new();

    if let Some(rec) = latest_by_name(state, "recommendation.ready") {
        lines.push(Line::from(format!(
            "summary: {}",
            truncate(payload_str(rec, "summary").unwrap_or("-"), 66)
        )));
        if let Some(manual) = payload_str(rec, "manual_citation") {
            lines.push(Line::from(format!("manual: {}", truncate(manual, 66))));
        }
        if let Some(action_count) = payload_u64(rec, "action_count") {
            lines.push(Line::from(format!("candidate count: {action_count}")));
        }
    } else {
        lines.push(Line::from("(no AI analysis yet)"));
    }

    lines.push(Line::from("-"));
    let max_rows = area.height.saturating_sub(2) as usize;
    let reserved = lines.len();
    let body_rows = max_rows.saturating_sub(reserved);
    let len = state.action_events.len();
    let start = clamped_index_window_start(state.actions_scroll_start, len, body_rows);
    let end = (start + body_rows).min(len);

    if len == 0 {
        lines.push(Line::from("(no analysis events)"));
    } else {
        for (i, event) in state
            .action_events
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            let selected = i == state.selected_actions;
            let style = if selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            lines.push(Line::styled(analysis_event_label(event), style));
        }
    }

    let p = Paragraph::new(Text::from(lines))
        .block(panel_block("Analysis Phase", focused));
    f.render_widget(p, area);
}

fn render_candidate_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Evidence;
    let block = panel_block("Candidate Runs", focused);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let runs = candidate_run_entries(state);
    if runs.is_empty() {
        let p = Paragraph::new("(no candidate runs yet)");
        f.render_widget(p, inner);
        return;
    }

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(inner);

    let bar_values: Vec<u64> = runs
        .iter()
        .map(|run| {
            let millis = (run.wall_time_s * 1000.0).round();
            if millis < 1.0 { 1 } else { millis as u64 }
        })
        .collect();
    let max_value = bar_values.iter().copied().max().unwrap_or(1);

    let bars: Vec<Bar<'static>> = runs
        .iter()
        .enumerate()
        .zip(bar_values.iter().copied())
        .map(|((idx, run), value)| {
            Bar::default()
                .label(format!("S{idx}").into())
                .value(value)
                .style(candidate_bar_style(run.round))
        })
        .collect();

    let chart = BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width(3)
        .bar_gap(1)
        .max(max_value)
        .bar_style(Style::default().fg(Color::Gray))
        .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
        .label_style(Style::default().fg(Color::Gray));
    f.render_widget(chart, sections[0]);

    let mut legend_lines = Vec::new();
    let legend_rows = sections[1].height as usize;
    let start = clamped_index_window_start(state.evidence_scroll_start, runs.len(), legend_rows);
    let end = (start + legend_rows).min(runs.len());
    let baseline_wall = runs
        .iter()
        .find(|run| run.round == 0)
        .map(|run| run.wall_time_s)
        .or_else(|| runs.first().map(|run| run.wall_time_s))
        .unwrap_or(0.0);
    for idx in start..end {
        let run = &runs[idx];
        let selected = idx == state.selected_evidence;
        let mut style = candidate_bar_style(run.round);
        if selected {
            style = style.add_modifier(Modifier::REVERSED);
        }
        let improvement_pct = improvement_pct_vs_baseline(run.wall_time_s, baseline_wall);
        let status_suffix = run
            .profiling_status
            .map(|status| format!(" | {status}"))
            .unwrap_or_default();
        legend_lines.push(Line::styled(
            format!(
                "S{idx} {:.2}s | {:+.1}% | round {} c{}{}",
                run.wall_time_s,
                improvement_pct,
                run.round,
                run.candidate_id,
                status_suffix
            ),
            style,
        ));
    }
    if legend_lines.is_empty() {
        legend_lines.push(Line::from("(no run labels)"));
    }

    let legend = Paragraph::new(Text::from(legend_lines));
    f.render_widget(legend, sections[1]);
}

#[derive(Debug, Clone)]
struct CandidateRunEntry {
    wall_time_s: f64,
    round: u64,
    candidate_id: u64,
    profiling_status: Option<&'static str>,
}

fn candidate_run_entries(state: &AppState) -> Vec<CandidateRunEntry> {
    let mut runs = Vec::new();
    let selected_candidates = selected_candidates_for_profiling(state);
    let (profiling_in_progress_round, completed_profiling_rounds) = profiling_state(state);

    for event in state.evidence_events.iter() {
        if event.event == "phase.finished" {
            let phase = payload_str(event, "phase").unwrap_or("");
            if !phase.contains("baseline") {
                continue;
            }
            if let Some(wall_time_s) = payload_f64(event, "wall_time_s") {
                runs.push(CandidateRunEntry {
                    wall_time_s,
                    round: 0,
                    candidate_id: 0,
                    profiling_status: candidate_profiling_status(
                        0,
                        None,
                        &selected_candidates,
                        profiling_in_progress_round,
                        &completed_profiling_rounds,
                    ),
                });
            }
            continue;
        }

        if event.event == "candidate.evaluated" {
            if let Some(wall_time_s) = payload_f64(event, "wall_time_s") {
                let round = payload_u64(event, "iteration").unwrap_or(0);
                let candidate_id = payload_u64(event, "candidate_id");
                runs.push(CandidateRunEntry {
                    wall_time_s,
                    round,
                    candidate_id: candidate_id.unwrap_or(0),
                    profiling_status: candidate_profiling_status(
                        round,
                        candidate_id,
                        &selected_candidates,
                        profiling_in_progress_round,
                        &completed_profiling_rounds,
                    ),
                });
            }
        }
    }

    runs
}

fn candidate_bar_style(round: u64) -> Style {
    if round == 0 {
        return Style::default().fg(Color::LightYellow);
    }

    if round % 2 == 1 {
        Style::default().fg(Color::LightBlue)
    } else {
        Style::default().fg(Color::LightGreen)
    }
}

fn improvement_pct_vs_baseline(wall_time_s: f64, baseline_wall_time_s: f64) -> f64 {
    if baseline_wall_time_s <= 0.0 {
        return 0.0;
    }
    ((baseline_wall_time_s - wall_time_s) / baseline_wall_time_s) * 100.0
}

fn selected_candidates_for_profiling(state: &AppState) -> HashMap<u64, u64> {
    let mut selected = HashMap::new();
    for event in state.all_events.iter() {
        if event.event != "candidate.evaluated" {
            continue;
        }
        if payload_str(event, "verdict") != Some("improved") {
            continue;
        }

        let round = payload_u64(event, "iteration").unwrap_or(0);
        let candidate_id = payload_u64(event, "candidate_id").unwrap_or(0);
        if round > 0 && candidate_id > 0 {
            selected.insert(round, candidate_id);
        }
    }
    selected
}

fn profiling_state(state: &AppState) -> (Option<u64>, HashSet<u64>) {
    let mut completed_rounds = HashSet::new();
    for event in state.all_events.iter() {
        if event.event != "profile.finished" {
            continue;
        }
        if let Some(round) = payload_u64(event, "iteration") {
            completed_rounds.insert(round);
        }
    }

    if completed_rounds.is_empty() {
        for event in state.all_events.iter() {
            if event.event != "profile.snapshot" {
                continue;
            }
            if let Some(round) = payload_u64(event, "iteration") {
                completed_rounds.insert(round);
            }
        }
    }

    let mut active_round = state
        .all_events
        .iter()
        .rev()
        .find(|event| event.event == "profile.started" || event.event == "profile.finished")
        .and_then(|event| {
            if event.event == "profile.started" {
                payload_u64(event, "iteration")
            } else {
                None
            }
        });

    if active_round.is_none() {
        active_round = state
            .all_events
            .iter()
            .rev()
            .find(|event| {
                (event.event == "exec.started" || event.event == "exec.finished")
                    && payload_str(event, "phase") == Some("spawn.profiling")
            })
            .and_then(|event| {
                if event.event == "exec.started" {
                    payload_u64(event, "iteration")
                } else {
                    None
                }
            });
    }

    (active_round, completed_rounds)
}

fn candidate_profiling_status(
    round: u64,
    candidate_id: Option<u64>,
    selected_candidates: &HashMap<u64, u64>,
    profiling_in_progress_round: Option<u64>,
    completed_profiling_rounds: &HashSet<u64>,
) -> Option<&'static str> {
    if round == 0 {
        if profiling_in_progress_round == Some(1) {
            return Some("under profiling");
        }
        return Some("selected for profiling");
    }

    let candidate_id = candidate_id?;
    if candidate_id == 0 {
        return None;
    }

    let selected_candidate_id = selected_candidates.get(&round)?;
    if *selected_candidate_id != candidate_id {
        return None;
    }

    let next_round = round + 1;
    if profiling_in_progress_round == Some(next_round) {
        return Some("under profiling");
    }
    if completed_profiling_rounds.contains(&next_round) {
        return Some("selected for profiling");
    }
    None
}

fn render_decision_and_details(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Details;
    let mut lines: Vec<Line<'static>> = Vec::new();

    if let Some(env) = details_event(state) {
        if is_log_details_event(env) {
            if let Some(message) = env.payload.get("message").and_then(|value| value.as_str()) {
                let normalized = humanize_json_quote_escapes(message).replace("\\n", "\n");
                for line in normalized.lines() {
                    lines.push(render_rich_log_line(line, false));
                }
            } else {
                for line in pretty_payload_lines_without_outer_braces(&env.payload) {
                    lines.push(Line::from(line));
                }
            }
        } else {
            if is_progress_event(state, env) {
                lines.push(Line::from(format!("timeline: {}", run_event_label(env))));
                let wall_time_display = event_wall_time_info(state, env)
                    .map(|(value, _source)| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".to_string());
                let offset_display = event_time_offset_s(state, env)
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".to_string());
                lines.push(Line::from(format!(
                    "wall_time_s: {}",
                    wall_time_display
                )));
                lines.push(Line::from(format!("time_offset_s: {}", offset_display)));
                lines.push(Line::from("-"));
            }
            for line in pretty_payload_lines_without_outer_braces(&env.payload) {
                lines.push(render_rich_log_line(&line, false));
            }
        }
    } else {
        lines.push(Line::from("(no payload selected)"));
    }

    let scroll_y: u16 = state.details_scroll.min(u16::MAX as usize) as u16;

    let title = if state.pinned_seq.is_some() {
        "Decision / Details (pinned)"
    } else {
        "Decision / Details"
    };
    let p = Paragraph::new(Text::from(lines))
        .block(panel_block(title, focused))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(p, area);
}

fn render_logs(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Logs;
    let mut text = Text::default();
    let content_width = area.width.saturating_sub(2).max(1) as usize;
    let max_visible_chars = content_width.saturating_mul(6).max(180);

    let display_messages: Vec<String> = state
        .log_lines
        .iter()
        .map(|line| {
            let normalized = humanize_json_quote_escapes(&line.message);
            clip_log_message(&normalized, max_visible_chars)
        })
        .collect();

    for (i, message) in display_messages.iter().enumerate() {
        let selected = i == state.selected_logs;
        text.lines.push(render_rich_log_line(message, selected));
    }

    if text.lines.is_empty() {
        text.lines.push(Line::from("(no logs)"));
    }

    let max_scroll =
        max_wrapped_scroll_lines(&display_messages, content_width, area.height.saturating_sub(2) as usize);
    let scroll_y = state
        .logs_scroll_lines
        .min(max_scroll)
        .min(u16::MAX as usize) as u16;

    let p = Paragraph::new(text)
        .block(panel_block("Logs", focused))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(p, area);
}

fn details_event(state: &AppState) -> Option<&TuneEventEnvelope> {
    if let Some(seq) = state.pinned_seq {
        return state.find_by_seq(seq);
    }
    if state.focused == Panel::Details {
        return preferred_details_event(state);
    }

    details_event_for_panel(state, state.focused).or_else(|| state.last_event())
}

fn preferred_details_event(state: &AppState) -> Option<&TuneEventEnvelope> {
    let contextual = details_event_for_panel(state, state.last_non_details_focus);
    if state.last_non_details_focus == Panel::Logs {
        return latest_non_log_event(state)
            .or(contextual)
            .or_else(|| state.last_event());
    }

    contextual
        .or_else(|| latest_non_log_event(state))
        .or_else(|| state.last_event())
}

fn details_event_for_panel(state: &AppState, panel: Panel) -> Option<&TuneEventEnvelope> {
    match panel {
        Panel::Overview => latest_by_name(state, "run.started").or_else(|| state.last_event()),
        Panel::Progress => state.progress_events.get(state.selected_progress),
        Panel::Metrics => state.profiling_events.get(state.selected_profiling),
        Panel::Actions => state.action_events.get(state.selected_actions),
        Panel::Evidence => state.evidence_events.get(state.selected_evidence),
        Panel::Logs => state
            .log_lines
            .get(state.selected_logs)
            .and_then(|line| if line.seq == 0 { None } else { Some(line.seq) })
            .and_then(|seq| state.find_by_seq(seq)),
        _ => state.last_event(),
    }
}

fn run_event_label(event: &TuneEventEnvelope) -> String {
    match event.event.as_str() {
        "run.started" => "round 0 step 0 baseline run start".to_string(),
        "baseline.finished" => "round 0 step 0 baseline run finish".to_string(),
        "profile.started" => {
            let round = round_for_event(event);
            format!("round {round} step 1 profiling start")
        }
        "profile.finished" => {
            let round = round_for_event(event);
            format!("round {round} step 1 profiling finish")
        }
        "recommendation.ready" => {
            let round = round_for_event(event);
            format!("round {round} step 2 analysis")
        }
        "progress.update" => {
            let round = round_for_event(event);
            let current_trial = payload_u64(event, "current_trial").unwrap_or(0);
            let total_trials = payload_u64(event, "total_trials").unwrap_or(0);
            if current_trial > 0 && total_trials > 0 {
                format!("round {round} step 3 candidate search {current_trial}/{total_trials}")
            } else {
                format!("round {round} step 3 candidate search")
            }
        }
        "progress.finished" => {
            let round = round_for_event(event);
            let current_trial = payload_u64(event, "current_trial").unwrap_or(0);
            let total_trials = payload_u64(event, "total_trials").unwrap_or(0);
            if current_trial > 0 && total_trials > 0 {
                format!(
                    "round {round} step 3 candidate search {current_trial}/{total_trials} finish"
                )
            } else {
                format!("round {round} step 3 candidate search finish")
            }
        }
        "candidate.evaluated" => {
            let round = round_for_event(event);
            let candidate_id = payload_u64(event, "candidate_id").unwrap_or(0);
            if candidate_id > 0 {
                format!("round {round} step 3 candidate {candidate_id} evaluated")
            } else {
                format!("round {round} step 3 candidate evaluated")
            }
        }
        "result.comparison" => "round final step 4 results comparison".to_string(),
        _ => event.event.clone(),
    }
}

fn analysis_event_label(event: &TuneEventEnvelope) -> String {
    let round = payload_u64(event, "iteration").unwrap_or(0);
    if event.event == "recommendation.ready" {
        return format!("round {round} recommendation");
    }
    if event.event.starts_with("recommendation.") {
        return format!("round {round} {}", event.event.replace("recommendation.", "recommendation "));
    }
    if event.event.starts_with("action.") {
        return format!("round {round} {}", event.event.replace("action.", "action "));
    }
    format!("round {round} {}", event.event)
}

fn round_for_event(event: &TuneEventEnvelope) -> u64 {
    if let Some(iteration) = payload_u64(event, "iteration") {
        return iteration;
    }

    let phase = payload_str(event, "phase").unwrap_or("");
    if phase.contains("baseline") {
        return 0;
    }
    0
}

fn is_progress_event(state: &AppState, event: &TuneEventEnvelope) -> bool {
    state.progress_events.iter().any(|entry| entry.seq == event.seq)
}

fn latest_by_name<'a>(state: &'a AppState, name: &str) -> Option<&'a TuneEventEnvelope> {
    state.all_events.iter().rev().find(|event| event.event == name)
}

fn latest_non_log_event(state: &AppState) -> Option<&TuneEventEnvelope> {
    state
        .all_events
        .iter()
        .rev()
        .find(|event| event.event_type != "log" && !event.event.starts_with("log."))
}

fn app_execution_running(state: &AppState) -> bool {
    state
        .all_events
        .iter()
        .rev()
        .find(|event| event.event == "exec.started" || event.event == "exec.finished")
        .map(|event| event.event == "exec.started")
        .unwrap_or(false)
}

fn app_execution_spinner(running: bool) -> char {
    if !running {
        return 'o';
    }

    const FRAMES: [char; 4] = ['|', '/', '-', '\\'];
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let frame = ((millis / 120) as usize) % FRAMES.len();
    FRAMES[frame]
}

fn payload_str<'a>(event: &'a TuneEventEnvelope, key: &str) -> Option<&'a str> {
    event.payload.get(key).and_then(|value| value.as_str())
}

fn payload_str_list(event: &TuneEventEnvelope, key: &str) -> Vec<String> {
    event
        .payload
        .get(key)
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .map(|value| value.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn payload_f64(event: &TuneEventEnvelope, key: &str) -> Option<f64> {
    event.payload.get(key).and_then(|value| value.as_f64())
}

fn payload_u64(event: &TuneEventEnvelope, key: &str) -> Option<u64> {
    event.payload.get(key).and_then(|value| value.as_u64())
}

fn event_wall_time_info(state: &AppState, event: &TuneEventEnvelope) -> Option<(f64, &'static str)> {
    if let Some(value) = payload_f64(event, "wall_time_s") {
        return Some((value, "payload"));
    }

    if let Some(value) = event
        .payload
        .get("metrics")
        .and_then(|metrics| metrics.get("wall_time_s"))
        .and_then(|value| value.as_f64())
    {
        return Some((value, "metrics"));
    }

    if event.event == "progress.update" || event.event.ends_with(".started") {
        return Some((0.0, "marker"));
    }

    if let Some(value) = derived_phase_wall_time_s(state, event) {
        return Some((value, "derived"));
    }

    event_time_offset_s(state, event).map(|value| (value, "counter"))
}

fn derived_phase_wall_time_s(state: &AppState, event: &TuneEventEnvelope) -> Option<f64> {
    match event.event.as_str() {
        "phase.finished" => {
            let start = state
                .all_events
                .iter()
                .rev()
                .find(|candidate| {
                    candidate.seq < event.seq
                        && candidate.event == "phase.started"
                        && same_phase_key(candidate, event)
                })?;
            duration_between_events(start, event)
        }
        "phase.started" => {
            let finish = state
                .all_events
                .iter()
                .find(|candidate| {
                    candidate.seq > event.seq
                        && candidate.event == "phase.finished"
                        && same_phase_key(candidate, event)
                })?;
            duration_between_events(event, finish)
        }
        "run.finished" => {
            let start = state
                .all_events
                .iter()
                .find(|candidate| candidate.event == "run.started")?;
            duration_between_events(start, event)
        }
        "run.started" => {
            let finish = state
                .all_events
                .iter()
                .find(|candidate| candidate.event == "run.finished")?;
            duration_between_events(event, finish)
        }
        _ => None,
    }
}

fn event_time_offset_s(state: &AppState, event: &TuneEventEnvelope) -> Option<f64> {
    let run_start = state
        .all_events
        .iter()
        .find(|candidate| candidate.event == "run.started")?;
    duration_between_events(run_start, event)
}

fn same_phase_key(lhs: &TuneEventEnvelope, rhs: &TuneEventEnvelope) -> bool {
    payload_str(lhs, "phase") == payload_str(rhs, "phase")
        && payload_u64(lhs, "iteration").unwrap_or(0)
            == payload_u64(rhs, "iteration").unwrap_or(0)
}

fn duration_between_events(start: &TuneEventEnvelope, end: &TuneEventEnvelope) -> Option<f64> {
    let start_s = parse_rfc3339_seconds(&start.ts)?;
    let end_s = parse_rfc3339_seconds(&end.ts)?;
    let diff = end_s - start_s;
    if diff.is_sign_negative() {
        None
    } else {
        Some(diff)
    }
}

fn parse_rfc3339_seconds(ts: &str) -> Option<f64> {
    let dt = DateTime::parse_from_rfc3339(ts).ok()?;
    Some((dt.timestamp_millis() as f64) / 1000.0)
}

fn render_rich_log_line(message: &str, selected: bool) -> Line<'static> {
    let base = if selected {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default()
    };

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut stack = vec![base];
    let mut current = String::new();
    let chars: Vec<char> = message.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        if chars[idx] == '[' {
            let mut end = idx + 1;
            while end < chars.len() && chars[end] != ']' {
                end += 1;
            }
            if end < chars.len() {
                let tag: String = chars[idx + 1..end].iter().collect();
                if let Some(stripped) = tag.strip_prefix('/') {
                    if is_supported_close_tag(stripped) {
                        flush_log_span(&mut spans, &mut current, *stack.last().unwrap_or(&base));
                        if stack.len() > 1 {
                            stack.pop();
                        }
                        idx = end + 1;
                        continue;
                    }
                } else if let Some(next_style) = parse_rich_tag(*stack.last().unwrap_or(&base), &tag) {
                    flush_log_span(&mut spans, &mut current, *stack.last().unwrap_or(&base));
                    stack.push(next_style);
                    idx = end + 1;
                    continue;
                }
            }
        }

        current.push(chars[idx]);
        idx += 1;
    }

    flush_log_span(&mut spans, &mut current, *stack.last().unwrap_or(&base));
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    Line::from(spans)
}

fn clip_log_message(message: &str, max_visible: usize) -> String {
    if max_visible == 0 {
        return "…".to_string();
    }

    let chars: Vec<char> = message.chars().collect();
    let mut output = String::new();
    let mut visible = 0usize;
    let mut idx = 0usize;

    while idx < chars.len() {
        if chars[idx] == '[' {
            let mut end = idx + 1;
            while end < chars.len() && chars[end] != ']' {
                end += 1;
            }
            if end < chars.len() {
                let tag: String = chars[idx + 1..end].iter().collect();
                if let Some(stripped) = tag.strip_prefix('/') {
                    if is_supported_close_tag(stripped) {
                        output.extend(chars[idx..=end].iter());
                        idx = end + 1;
                        continue;
                    }
                } else if parse_rich_tag(Style::default(), &tag).is_some() {
                    output.extend(chars[idx..=end].iter());
                    idx = end + 1;
                    continue;
                }
            }
        }

        if visible >= max_visible {
            break;
        }

        output.push(chars[idx]);
        visible += 1;
        idx += 1;
    }

    if idx < chars.len() {
        output.push('…');
    }

    output
}

fn humanize_json_quote_escapes(message: &str) -> String {
    let mut normalized = message.to_string();
    while normalized.contains("\\\"") {
        normalized = normalized.replace("\\\"", "\"");
    }
    normalized
}

fn flush_log_span(spans: &mut Vec<Span<'static>>, current: &mut String, style: Style) {
    if current.is_empty() {
        return;
    }
    spans.push(Span::styled(current.clone(), style));
    current.clear();
}

fn is_supported_close_tag(tag: &str) -> bool {
    parse_rich_tag(Style::default(), tag).is_some()
}

fn parse_rich_tag(base: Style, tag: &str) -> Option<Style> {
    let mut style = base;
    let mut matched = false;

    for token in tag.split_whitespace() {
        let normalized = token.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "bold" => {
                style = style.add_modifier(Modifier::BOLD);
                matched = true;
            }
            "dim" => {
                style = style.add_modifier(Modifier::DIM);
                matched = true;
            }
            "italic" => {
                style = style.add_modifier(Modifier::ITALIC);
                matched = true;
            }
            "underline" => {
                style = style.add_modifier(Modifier::UNDERLINED);
                matched = true;
            }
            "black" => {
                style = style.fg(Color::Black);
                matched = true;
            }
            "red" => {
                style = style.fg(Color::Red);
                matched = true;
            }
            "green" => {
                style = style.fg(Color::Green);
                matched = true;
            }
            "yellow" => {
                style = style.fg(Color::Yellow);
                matched = true;
            }
            "blue" => {
                style = style.fg(Color::Blue);
                matched = true;
            }
            "magenta" => {
                style = style.fg(Color::Magenta);
                matched = true;
            }
            "cyan" => {
                style = style.fg(Color::Cyan);
                matched = true;
            }
            "white" => {
                style = style.fg(Color::White);
                matched = true;
            }
            "bright_black" | "grey" | "gray" => {
                style = style.fg(Color::Gray);
                matched = true;
            }
            "bright_red" => {
                style = style.fg(Color::LightRed);
                matched = true;
            }
            "bright_green" => {
                style = style.fg(Color::LightGreen);
                matched = true;
            }
            "bright_yellow" => {
                style = style.fg(Color::LightYellow);
                matched = true;
            }
            "bright_blue" => {
                style = style.fg(Color::LightBlue);
                matched = true;
            }
            "bright_magenta" => {
                style = style.fg(Color::LightMagenta);
                matched = true;
            }
            "bright_cyan" => {
                style = style.fg(Color::LightCyan);
                matched = true;
            }
            "bright_white" => {
                style = style.fg(Color::White);
                matched = true;
            }
            _ => {}
        }
    }

    if matched {
        Some(style)
    } else {
        None
    }
}

fn wrapped_line_count(message: &str, width: usize) -> usize {
    let width = width.max(1);
    let plain = strip_supported_rich_tags(message);
    let mut count = 0usize;

    for segment in plain.split('\n') {
        let len = segment.chars().count();
        let rows = if len == 0 { 1 } else { ((len - 1) / width) + 1 };
        count += rows;
    }

    count.max(1)
}

fn max_wrapped_scroll_lines(messages: &[String], width: usize, view_height: usize) -> usize {
    let total_rows: usize = messages
        .iter()
        .map(|message| wrapped_line_count(message, width))
        .sum();
    total_rows.saturating_sub(view_height)
}

fn strip_supported_rich_tags(message: &str) -> String {
    let chars: Vec<char> = message.chars().collect();
    let mut output = String::new();
    let mut idx = 0usize;

    while idx < chars.len() {
        if chars[idx] == '[' {
            let mut end = idx + 1;
            while end < chars.len() && chars[end] != ']' {
                end += 1;
            }

            if end < chars.len() {
                let tag: String = chars[idx + 1..end].iter().collect();
                if let Some(stripped) = tag.strip_prefix('/') {
                    if is_supported_close_tag(stripped) {
                        idx = end + 1;
                        continue;
                    }
                } else if parse_rich_tag(Style::default(), &tag).is_some() {
                    idx = end + 1;
                    continue;
                }
            }
        }

        output.push(chars[idx]);
        idx += 1;
    }

    output
}

fn truncate(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_string();
    }
    format!("{}...", &value[..limit.saturating_sub(3)])
}

fn pretty_payload_lines_without_outer_braces(payload: &serde_json::Value) -> Vec<String> {
    let pretty = serde_json::to_string_pretty(payload)
        .unwrap_or_else(|_| "{}".to_string())
        .replace("\\n", "\n");

    let raw_lines: Vec<String> = pretty.lines().map(str::to_string).collect();
    if raw_lines.is_empty() {
        return vec!["(empty payload)".to_string()];
    }

    let first_is_open = raw_lines.first().map(|line| line.trim()) == Some("{");
    let last_is_close = raw_lines.last().map(|line| line.trim()) == Some("}");
    if first_is_open && last_is_close {
        let inner: Vec<String> = raw_lines[1..raw_lines.len().saturating_sub(1)].to_vec();
        if inner.is_empty() {
            return vec!["(empty payload)".to_string()];
        }

        let base_indent = inner
            .iter()
            .find(|line| !line.trim().is_empty())
            .map(|line| line.chars().take_while(|ch| *ch == ' ').count())
            .unwrap_or(0);

        return inner
            .into_iter()
            .map(|line| trim_leading_spaces(&line, base_indent))
            .collect();
    }

    raw_lines
}

fn is_log_details_event(event: &TuneEventEnvelope) -> bool {
    event.event_type == "log" || event.event.starts_with("log.")
}

fn trim_leading_spaces(line: &str, count: usize) -> String {
    let mut view = line;
    let mut remaining = count;

    while remaining > 0 {
        if let Some(rest) = view.strip_prefix(' ') {
            view = rest;
            remaining -= 1;
        } else {
            break;
        }
    }

    view.to_string()
}

fn clamped_index_window_start(start: usize, len: usize, max_rows: usize) -> usize {
    if len == 0 || max_rows == 0 || len <= max_rows {
        return 0;
    }
    start.min(len - max_rows)
}
