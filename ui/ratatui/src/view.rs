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
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(area);

    let top_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
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

    lines.push(Line::from(format!(
        "run_id: {}",
        state.run_id.as_deref().unwrap_or("-")
    )));
    lines.push(Line::from(format!(
        "schema: {}",
        state.schema_version.as_deref().unwrap_or("-")
    )));

    let run_started = latest_by_name(state, "run.started");
    let run_finished = latest_by_name(state, "run.finished");
    let phase_event = latest_phase_event(state);

    if let Some(started) = run_started {
        lines.push(Line::from(format!(
            "mode: {}",
            payload_str(started, "mode").unwrap_or("-")
        )));
        let cmd = payload_str(started, "command").unwrap_or("-");
        lines.push(Line::from(format!("cmd: {}", truncate(cmd, 52))));
    }

    if let Some(phase) = phase_event {
        let phase_name = payload_str(phase, "phase").unwrap_or("-");
        lines.push(Line::from(format!("phase: {}", phase_name)));
    }

    if let Some(done) = run_finished {
        let status = payload_str(done, "status").unwrap_or("-");
        lines.push(Line::from(format!("state: {status}")));
    } else {
        lines.push(Line::from("state: running"));
    }

    let p = Paragraph::new(Text::from(lines))
        .block(panel_block("Loop State", focused))
        .wrap(Wrap { trim: true });
    f.render_widget(p, area);
}

fn render_run_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Progress;
    let mut lines = Vec::new();

    if let Some(started) = latest_by_name(state, "run.started") {
        let cmd = payload_str(started, "command").unwrap_or("-");
        lines.push(Line::from(format!("running: {}", truncate(cmd, 66))));
        let timeout = payload_u64(started, "timeout_s").unwrap_or(0);
        if timeout > 0 {
            lines.push(Line::from(format!("timeout: {timeout}s")));
        }
    }

    if let Some(base) = latest_baseline_result(state) {
        if let Some(wall) = payload_f64(base, "wall_time_s") {
            lines.push(Line::from(format!("baseline wall: {wall:.2}s")));
        }
    }

    if let Some(result) = latest_by_name(state, "result.comparison") {
        let baseline = payload_f64(result, "baseline_time_s").unwrap_or(0.0);
        let tuned = payload_f64(result, "tuned_time_s").unwrap_or(0.0);
        let improvement = payload_f64(result, "improvement_pct").unwrap_or(0.0);
        lines.push(Line::from(format!("result: {baseline:.2}s -> {tuned:.2}s")));
        lines.push(Line::from(format!("improvement: {improvement:+.1}%")));
    }

    if let Some((round, stage)) = current_round_and_stage(state) {
        lines.push(Line::from(format!("current: round {round} {stage}")));
    }

    lines.push(Line::from("-"));

    let max_rows = area.height.saturating_sub(2) as usize;
    let reserved = lines.len();
    let body_rows = max_rows.saturating_sub(reserved);
    let len = state.progress_events.len();
    let (start, end) = visible_window(len, state.selected_progress, body_rows);

    if len == 0 {
        lines.push(Line::from("(no run events)"));
    } else {
        for (i, event) in state
            .progress_events
            .iter()
            .enumerate()
            .skip(start)
            .take(end - start)
        {
            let selected = i == state.selected_progress;
            let style = if selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };

            let mut suffix = String::new();
            if let Some(status) = payload_str(event, "status") {
                suffix = format!(" [{status}]");
            }
            lines.push(Line::styled(format!("{}{}", run_event_label(event), suffix), style));
        }
    }

    let p = Paragraph::new(Text::from(lines))
        .block(panel_block("Run Phase", focused));
    f.render_widget(p, area);
}

fn render_profiling_phase(f: &mut Frame, area: Rect, state: &AppState) {
    let focused = state.focused == Panel::Metrics;
    let mut lines = Vec::new();

    if state.profiling_events.is_empty() {
        lines.push(Line::from("(no profiling snapshot yet)"));
    } else {
        let max_rows = area.height.saturating_sub(2) as usize;
        let len = state.profiling_events.len();
        let (start, end) = visible_window(len, state.selected_profiling, max_rows);

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
    let (start, end) = visible_window(len, state.selected_actions, body_rows);

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
    let (start, end) = visible_window(runs.len(), state.selected_evidence, legend_rows);
    for idx in start..end {
        let run = &runs[idx];
        let selected = idx == state.selected_evidence;
        let mut style = candidate_bar_style(run.round);
        if selected {
            style = style.add_modifier(Modifier::REVERSED);
        }
        legend_lines.push(Line::styled(
            format!(
                "S{idx} {:.2}s | round {} | {}",
                run.wall_time_s,
                run.round,
                truncate(&run.cmd_diff, 34)
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
    cmd_diff: String,
    round: u64,
}

fn candidate_run_entries(state: &AppState) -> Vec<CandidateRunEntry> {
    let mut runs = Vec::new();

    for event in state.evidence_events.iter() {
        if event.event == "phase.finished" {
            let phase = payload_str(event, "phase").unwrap_or("");
            if !phase.contains("baseline") {
                continue;
            }
            if let Some(wall_time_s) = payload_f64(event, "wall_time_s") {
                let cmd_diff = payload_str(event, "cmd_diff")
                    .unwrap_or("baseline")
                    .to_string();
                runs.push(CandidateRunEntry {
                    wall_time_s,
                    cmd_diff,
                    round: 0,
                });
            }
            continue;
        }

        if event.event == "candidate.evaluated" {
            if let Some(wall_time_s) = payload_f64(event, "wall_time_s") {
                let cmd_diff = payload_str(event, "cmd_diff")
                    .or_else(|| payload_str(event, "label"))
                    .unwrap_or("(candidate)")
                    .to_string();
                runs.push(CandidateRunEntry {
                    wall_time_s,
                    cmd_diff,
                    round: payload_u64(event, "iteration").unwrap_or(0),
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
            for line in pretty_payload_lines_without_outer_braces(&env.payload) {
                lines.push(Line::from(line));
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

    let selected_idx = if state.log_lines.is_empty() {
        0
    } else {
        state.selected_logs.min(state.log_lines.len().saturating_sub(1))
    };
    let scroll_lines: usize = display_messages
        .iter()
        .take(selected_idx)
        .map(|line| wrapped_line_count(line, content_width))
        .sum();
    let scroll_y = scroll_lines.min(u16::MAX as usize) as u16;

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
        "run.started" => "round 0 baseline run start".to_string(),
        "run.finished" => "final run finished".to_string(),
        "phase.started" | "phase.finished" => {
            let round = round_for_event(event);
            let phase = payload_str(event, "phase").unwrap_or("phase");
            let state = if event.event == "phase.started" {
                "start"
            } else {
                "finish"
            };
            format!("round {round} {phase} {state}")
        }
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

fn current_round_and_stage(state: &AppState) -> Option<(u64, String)> {
    let event = state
        .all_events
        .iter()
        .rev()
        .find(|entry| entry.event.starts_with("phase.") || entry.event.starts_with("run."))?;

    if event.event == "run.finished" {
        return Some((round_for_event(event), "completed".to_string()));
    }

    if event.event == "run.started" {
        return Some((0, "baseline running".to_string()));
    }

    let round = round_for_event(event);
    let phase = payload_str(event, "phase").unwrap_or("phase");
    let state_label = if event.event == "phase.started" {
        "running"
    } else {
        "done"
    };
    Some((round, format!("{phase} {state_label}")))
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

fn latest_phase_event(state: &AppState) -> Option<&TuneEventEnvelope> {
    state
        .all_events
        .iter()
        .rev()
        .find(|event| event.event.starts_with("phase."))
}

fn latest_baseline_result(state: &AppState) -> Option<&TuneEventEnvelope> {
    state.all_events.iter().rev().find(|event| {
        event.event == "phase.finished"
            && payload_str(event, "phase")
                .map(|phase| phase.contains("baseline"))
                .unwrap_or(false)
    })
}

fn payload_str<'a>(event: &'a TuneEventEnvelope, key: &str) -> Option<&'a str> {
    event.payload.get(key).and_then(|value| value.as_str())
}

fn payload_f64(event: &TuneEventEnvelope, key: &str) -> Option<f64> {
    event.payload.get(key).and_then(|value| value.as_f64())
}

fn payload_u64(event: &TuneEventEnvelope, key: &str) -> Option<u64> {
    event.payload.get(key).and_then(|value| value.as_u64())
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

fn visible_window(len: usize, selected: usize, max_rows: usize) -> (usize, usize) {
    if len == 0 || max_rows == 0 || len <= max_rows {
        return (0, len);
    }
    let selected = selected.min(len - 1);
    let half = max_rows / 2;
    let start = selected.saturating_sub(half).min(len - max_rows);
    (start, start + max_rows)
}
