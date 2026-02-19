use std::{
    io::{self, BufRead},
    sync::mpsc::{self, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use ratatui::crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode, KeyEvent,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, size as terminal_size, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::{backend::CrosstermBackend, Terminal};

use crate::{
    event::parse_event_line,
    reducer::{reduce, AppMsg},
    state::{AppState, Panel},
    view::draw,
};

pub fn run() -> io::Result<()> {
    let mut tui = Tui::init()?;
    let (tx, rx) = mpsc::channel::<AppMsg>();
    spawn_stdin_reader(tx);

    let mut state = AppState::new();
    let mut mouse_tracker = MouseClickTracker::default();
    let tick_rate = Duration::from_millis(100);
    loop {
        loop {
            match rx.try_recv() {
                Ok(msg) => reduce(&mut state, msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        tui.terminal.draw(|f| draw(f, &state))?;

        if event::poll(tick_rate)? {
            match event::read()? {
                CEvent::Key(key) => {
                    if handle_key(&mut state, &mut tui, key) {
                        break;
                    }
                }
                CEvent::Mouse(mouse) => {
                    handle_mouse(&mut state, &mut mouse_tracker, mouse);
                }
                _ => {}
            }
        }
    }

    tui.restore()?;
    Ok(())
}

fn spawn_stdin_reader(tx: mpsc::Sender<AppMsg>) {
    thread::spawn(move || {
        let stdin = io::stdin();
        let lock = stdin.lock();
        for line in lock.lines() {
            match line {
                Ok(line) => match parse_event_line(&line) {
                    Ok(parsed) => {
                        if let Some(env) = parsed {
                            let _ = tx.send(AppMsg::StdinEvent(env));
                        }
                    }
                    Err(kind) => {
                        let _ = tx.send(AppMsg::StdinParseError { line, kind });
                    }
                },
                Err(_) => break,
            }
        }
        let _ = tx.send(AppMsg::StdinClosed);
    });
}

fn handle_key(state: &mut AppState, tui: &mut Tui, key: KeyEvent) -> bool {
    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), _) => return true,
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            return true
        }
        (KeyCode::F(2), _) => {
            let _ = tui.toggle_mouse_capture();
        }
        (KeyCode::Tab, KeyModifiers::NONE) => set_focus(state, state.focused.next()),
        (KeyCode::BackTab, _) | (KeyCode::Tab, KeyModifiers::SHIFT) => {
            set_focus(state, state.focused.prev())
        }
        (KeyCode::Up, _) => move_vertical(state, -1),
        (KeyCode::Down, _) => move_vertical(state, 1),
        (KeyCode::PageUp, _) => move_vertical(state, -8),
        (KeyCode::PageDown, _) => move_vertical(state, 8),
        (KeyCode::Enter, _) => toggle_pin(state),
        (KeyCode::Esc, _) => {
            state.pinned_seq = None;
            state.details_scroll = 0;
        }
        (KeyCode::Left, _) => set_focus(state, state.focused.prev()),
        (KeyCode::Right, _) => set_focus(state, state.focused.next()),
        _ => {}
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryPanel {
    Progress,
    Profiling,
    Analysis,
    Candidate,
    Logs,
}

#[derive(Debug, Clone, Copy)]
struct ClickSelection {
    panel: EntryPanel,
    index: usize,
}

#[derive(Debug, Clone, Copy)]
enum MouseTarget {
    Entries(EntryPanel),
    Details,
}

#[derive(Debug, Default)]
struct MouseClickTracker {
    last: Option<(EntryPanel, usize, Instant)>,
}

impl MouseClickTracker {
    fn register(&mut self, selection: ClickSelection) -> bool {
        let now = Instant::now();
        let is_double = self
            .last
            .map(|(panel, index, ts)| {
                panel == selection.panel
                    && index == selection.index
                    && now.duration_since(ts) <= Duration::from_millis(350)
            })
            .unwrap_or(false);
        self.last = Some((selection.panel, selection.index, now));
        is_double
    }
}

fn handle_mouse(state: &mut AppState, tracker: &mut MouseClickTracker, mouse: MouseEvent) {
    let hit = compute_hit_layout(state, current_area_rect());
    let target = target_at(&hit, mouse.column, mouse.row);

    match mouse.kind {
        MouseEventKind::Up(MouseButton::Left) => {
            if let Some(selection) = click_select_entry(state, &hit, target, mouse.column, mouse.row)
            {
                if tracker.register(selection) {
                    toggle_pin_for_entry(state, selection.panel);
                }
            } else if matches!(target, Some(MouseTarget::Details)) {
                set_focus(state, Panel::Details);
            }
        }
        MouseEventKind::ScrollUp => {
            if state.pinned_seq.is_some() {
                bump_scroll(&mut state.details_scroll, -3);
                return;
            }
            match target {
                Some(MouseTarget::Details) => {
                    set_focus(state, Panel::Details);
                    bump_scroll(&mut state.details_scroll, -3);
                }
                Some(MouseTarget::Entries(panel)) => {
                    set_focus(state, panel_to_view_panel(panel));
                    move_selection(state, -1);
                }
                _ => {}
            }
        }
        MouseEventKind::ScrollDown => {
            if state.pinned_seq.is_some() {
                bump_scroll(&mut state.details_scroll, 3);
                return;
            }
            match target {
                Some(MouseTarget::Details) => {
                    set_focus(state, Panel::Details);
                    bump_scroll(&mut state.details_scroll, 3);
                }
                Some(MouseTarget::Entries(panel)) => {
                    set_focus(state, panel_to_view_panel(panel));
                    move_selection(state, 1);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn toggle_pin_for_entry(state: &mut AppState, panel: EntryPanel) {
    let seq = selected_seq_for_panel(state, panel_to_view_panel(panel));
    if state.pinned_seq == seq {
        state.pinned_seq = None;
    } else {
        state.pinned_seq = seq;
    }
    state.details_scroll = 0;
}

fn panel_to_view_panel(panel: EntryPanel) -> Panel {
    match panel {
        EntryPanel::Progress => Panel::Progress,
        EntryPanel::Profiling => Panel::Metrics,
        EntryPanel::Analysis => Panel::Actions,
        EntryPanel::Candidate => Panel::Evidence,
        EntryPanel::Logs => Panel::Logs,
    }
}

#[derive(Debug, Clone, Copy)]
struct HitLayout {
    run_entries: Rect,
    profiling_entries: Rect,
    analysis_entries: Rect,
    candidate_entries: Rect,
    logs_entries: Rect,
    details: Rect,
}

fn compute_hit_layout(state: &AppState, area: Rect) -> HitLayout {
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

    let run_inner = inner_rect(left[1]);
    let run_reserved = run_reserved_lines(state) as u16;
    let run_entries = Rect {
        x: run_inner.x,
        y: run_inner.y.saturating_add(run_reserved),
        width: run_inner.width,
        height: run_inner.height.saturating_sub(run_reserved),
    };

    let analysis_inner = inner_rect(mid_top[0]);
    let analysis_reserved = analysis_reserved_lines(state) as u16;
    let analysis_entries = Rect {
        x: analysis_inner.x,
        y: analysis_inner.y.saturating_add(analysis_reserved),
        width: analysis_inner.width,
        height: analysis_inner.height.saturating_sub(analysis_reserved),
    };

    let profiling_entries = inner_rect(mid_top[1]);

    let candidate_inner = inner_rect(bottom[0]);
    let candidate_sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(candidate_inner);

    let logs_entries = inner_rect(bottom[1]);
    let details = inner_rect(top_cols[2]);

    HitLayout {
        run_entries,
        profiling_entries,
        analysis_entries,
        candidate_entries: candidate_sections[1],
        logs_entries,
        details,
    }
}

fn current_area_rect() -> Rect {
    let (width, height) = terminal_size().unwrap_or((120, 40));
    Rect {
        x: 0,
        y: 0,
        width,
        height,
    }
}

fn target_at(hit: &HitLayout, column: u16, row: u16) -> Option<MouseTarget> {
    if rect_contains(hit.run_entries, column, row) {
        return Some(MouseTarget::Entries(EntryPanel::Progress));
    }
    if rect_contains(hit.profiling_entries, column, row) {
        return Some(MouseTarget::Entries(EntryPanel::Profiling));
    }
    if rect_contains(hit.analysis_entries, column, row) {
        return Some(MouseTarget::Entries(EntryPanel::Analysis));
    }
    if rect_contains(hit.candidate_entries, column, row) {
        return Some(MouseTarget::Entries(EntryPanel::Candidate));
    }
    if rect_contains(hit.logs_entries, column, row) {
        return Some(MouseTarget::Entries(EntryPanel::Logs));
    }
    if rect_contains(hit.details, column, row) {
        return Some(MouseTarget::Details);
    }
    None
}

fn click_select_entry(
    state: &mut AppState,
    hit: &HitLayout,
    target: Option<MouseTarget>,
    column: u16,
    row: u16,
) -> Option<ClickSelection> {
    match target {
        Some(MouseTarget::Entries(EntryPanel::Progress)) => {
            let idx = row_to_window_index(
                row,
                hit.run_entries,
                state.progress_events.len(),
                state.selected_progress,
            )?;
            state.selected_progress = idx;
            set_focus(state, Panel::Progress);
            Some(ClickSelection {
                panel: EntryPanel::Progress,
                index: idx,
            })
        }
        Some(MouseTarget::Entries(EntryPanel::Profiling)) => {
            let idx = row_to_window_index(
                row,
                hit.profiling_entries,
                state.profiling_events.len(),
                state.selected_profiling,
            )?;
            state.selected_profiling = idx;
            set_focus(state, Panel::Metrics);
            Some(ClickSelection {
                panel: EntryPanel::Profiling,
                index: idx,
            })
        }
        Some(MouseTarget::Entries(EntryPanel::Analysis)) => {
            let idx = row_to_window_index(
                row,
                hit.analysis_entries,
                state.action_events.len(),
                state.selected_actions,
            )?;
            state.selected_actions = idx;
            set_focus(state, Panel::Actions);
            Some(ClickSelection {
                panel: EntryPanel::Analysis,
                index: idx,
            })
        }
        Some(MouseTarget::Entries(EntryPanel::Candidate)) => {
            let len = candidate_run_len(state);
            let idx = row_to_window_index(
                row,
                hit.candidate_entries,
                len,
                state.selected_evidence.min(len.saturating_sub(1)),
            )?;
            state.selected_evidence = idx;
            set_focus(state, Panel::Evidence);
            Some(ClickSelection {
                panel: EntryPanel::Candidate,
                index: idx,
            })
        }
        Some(MouseTarget::Entries(EntryPanel::Logs)) => {
            let idx = row_to_logs_index(state, hit.logs_entries, row, column)?;
            state.selected_logs = idx;
            set_focus(state, Panel::Logs);
            Some(ClickSelection {
                panel: EntryPanel::Logs,
                index: idx,
            })
        }
        _ => None,
    }
}

fn row_to_window_index(
    row: u16,
    rect: Rect,
    len: usize,
    selected: usize,
) -> Option<usize> {
    if len == 0 || rect.height == 0 || !rect_contains(rect, rect.x, row) {
        return None;
    }
    let y_offset = row.saturating_sub(rect.y) as usize;
    let max_rows = rect.height as usize;
    let (start, end) = visible_window(len, selected.min(len.saturating_sub(1)), max_rows);
    let visible_len = end.saturating_sub(start);
    if y_offset >= visible_len {
        return None;
    }
    Some(start + y_offset)
}

fn row_to_logs_index(state: &AppState, rect: Rect, row: u16, _column: u16) -> Option<usize> {
    if state.log_lines.is_empty() || rect.height == 0 || !rect_contains(rect, rect.x, row) {
        return None;
    }

    let content_width = rect.width.max(1) as usize;
    let max_visible_chars = content_width.saturating_mul(6).max(180);
    let messages: Vec<String> = state
        .log_lines
        .iter()
        .map(|line| clip_log_message_for_hit(&humanize_json_quote_escapes_for_hit(&line.message), max_visible_chars))
        .collect();

    let selected_idx = state.selected_logs.min(messages.len().saturating_sub(1));
    let base_scroll_lines: usize = messages
        .iter()
        .take(selected_idx)
        .map(|line| wrapped_line_count_for_hit(line, content_width))
        .sum();

    let absolute_line = base_scroll_lines + row.saturating_sub(rect.y) as usize;
    let mut acc = 0usize;
    for (idx, message) in messages.iter().enumerate() {
        let rows = wrapped_line_count_for_hit(message, content_width);
        if absolute_line < acc + rows {
            return Some(idx);
        }
        acc += rows;
    }

    Some(messages.len().saturating_sub(1))
}

fn candidate_run_len(state: &AppState) -> usize {
    state
        .evidence_events
        .iter()
        .filter(|event| {
            if event.event == "phase.finished" {
                let phase = event
                    .payload
                    .get("phase")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                return phase.contains("baseline")
                    && event.payload.get("wall_time_s").and_then(|v| v.as_f64()).is_some();
            }
            event.event == "candidate.evaluated"
                && event.payload.get("wall_time_s").and_then(|v| v.as_f64()).is_some()
        })
        .count()
}

fn run_reserved_lines(state: &AppState) -> usize {
    let mut reserved = 0usize;
    if latest_event_by_name(state, "run.started").is_some() {
        reserved += 1;
        if latest_event_by_name(state, "run.started")
            .and_then(|event| event.payload.get("timeout_s").and_then(|v| v.as_u64()))
            .unwrap_or(0)
            > 0
        {
            reserved += 1;
        }
    }
    if latest_baseline_event(state).is_some() {
        reserved += 1;
    }
    if latest_event_by_name(state, "result.comparison").is_some() {
        reserved += 2;
    }
    if current_round_and_stage_for_hit(state).is_some() {
        reserved += 1;
    }
    reserved + 1
}

fn analysis_reserved_lines(state: &AppState) -> usize {
    let mut reserved = 0usize;
    if let Some(rec) = latest_event_by_name(state, "recommendation.ready") {
        let _ = rec;
        reserved += 1;
        if latest_event_by_name(state, "recommendation.ready")
            .and_then(|event| event.payload.get("manual_citation").and_then(|v| v.as_str()))
            .is_some()
        {
            reserved += 1;
        }
        if latest_event_by_name(state, "recommendation.ready")
            .and_then(|event| event.payload.get("action_count").and_then(|v| v.as_u64()))
            .is_some()
        {
            reserved += 1;
        }
    } else {
        reserved += 1;
    }
    reserved + 1
}

fn latest_event_by_name<'a>(state: &'a AppState, name: &str) -> Option<&'a crate::event::TuneEventEnvelope> {
    state.all_events.iter().rev().find(|event| event.event == name)
}

fn latest_baseline_event<'a>(state: &'a AppState) -> Option<&'a crate::event::TuneEventEnvelope> {
    state.all_events.iter().rev().find(|event| {
        event.event == "phase.finished"
            && event
                .payload
                .get("phase")
                .and_then(|v| v.as_str())
                .map(|phase| phase.contains("baseline"))
                .unwrap_or(false)
    })
}

fn current_round_and_stage_for_hit(state: &AppState) -> Option<(u64, String)> {
    let event = state
        .all_events
        .iter()
        .rev()
        .find(|entry| entry.event.starts_with("phase.") || entry.event.starts_with("run."))?;
    if event.event == "run.finished" {
        return Some((0, "completed".to_string()));
    }
    if event.event == "run.started" {
        return Some((0, "baseline running".to_string()));
    }
    let round = event
        .payload
        .get("iteration")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let phase = event
        .payload
        .get("phase")
        .and_then(|v| v.as_str())
        .unwrap_or("phase");
    let state_label = if event.event == "phase.started" {
        "running"
    } else {
        "done"
    };
    Some((round, format!("{phase} {state_label}")))
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

fn inner_rect(rect: Rect) -> Rect {
    if rect.width <= 2 || rect.height <= 2 {
        return Rect {
            x: rect.x,
            y: rect.y,
            width: 0,
            height: 0,
        };
    }
    Rect {
        x: rect.x + 1,
        y: rect.y + 1,
        width: rect.width - 2,
        height: rect.height - 2,
    }
}

fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
    column >= rect.x
        && column < rect.x.saturating_add(rect.width)
        && row >= rect.y
        && row < rect.y.saturating_add(rect.height)
}

fn humanize_json_quote_escapes_for_hit(message: &str) -> String {
    let mut normalized = message.to_string();
    while normalized.contains("\\\"") {
        normalized = normalized.replace("\\\"", "\"");
    }
    normalized
}

fn clip_log_message_for_hit(message: &str, max_visible: usize) -> String {
    if max_visible == 0 {
        return "…".to_string();
    }

    let mut output = String::new();
    let mut visible = 0usize;
    for ch in message.chars() {
        if visible >= max_visible {
            output.push('…');
            break;
        }
        output.push(ch);
        visible += 1;
    }
    output
}

fn wrapped_line_count_for_hit(message: &str, width: usize) -> usize {
    let width = width.max(1);
    let mut count = 0usize;
    for segment in message.split('\n') {
        let len = segment.chars().count();
        count += if len == 0 { 1 } else { ((len - 1) / width) + 1 };
    }
    count.max(1)
}

fn set_focus(state: &mut AppState, next: Panel) {
    state.focused = next;
    if next != Panel::Details {
        state.last_non_details_focus = next;
    }
}

fn move_vertical(state: &mut AppState, delta: i32) {
    if state.pinned_seq.is_some() {
        bump_scroll(&mut state.details_scroll, delta);
        return;
    }
    move_selection(state, delta);
}

fn move_selection(state: &mut AppState, delta: i32) {
    let d = delta;
    match state.focused {
        Panel::Progress => bump(&mut state.selected_progress, state.progress_events.len(), d),
        Panel::Metrics => bump(&mut state.selected_profiling, state.profiling_events.len(), d),
        Panel::Actions => bump(&mut state.selected_actions, state.action_events.len(), d),
        Panel::Evidence => bump(&mut state.selected_evidence, state.evidence_events.len(), d),
        Panel::Logs => bump(&mut state.selected_logs, state.log_lines.len(), d),
        Panel::Details => bump_scroll(&mut state.details_scroll, d),
        _ => {}
    }
}

fn bump_scroll(scroll: &mut usize, delta: i32) {
    if delta < 0 {
        *scroll = scroll.saturating_sub((-delta) as usize);
    } else {
        *scroll = scroll.saturating_add(delta as usize);
    }
}

fn bump(idx: &mut usize, len: usize, delta: i32) {
    if len == 0 {
        *idx = 0;
        return;
    }
    let next = (*idx as i32 + delta).clamp(0, (len - 1) as i32) as usize;
    *idx = next;
}

fn toggle_pin(state: &mut AppState) {
    if state.pinned_seq.is_some() {
        state.pinned_seq = None;
        state.details_scroll = 0;
        return;
    }

    let seq = match state.focused {
        Panel::Details => preferred_details_seq(state),
        panel => selected_seq_for_panel(state, panel),
    };
    state.pinned_seq = seq;
    state.details_scroll = 0;
}

fn preferred_details_seq(state: &AppState) -> Option<u64> {
    let contextual = selected_seq_for_panel(state, state.last_non_details_focus);
    if state.last_non_details_focus == Panel::Logs {
        return latest_non_log_seq(state)
            .or(contextual)
            .or_else(|| state.last_event().map(|event| event.seq));
    }

    contextual
        .or_else(|| latest_non_log_seq(state))
        .or_else(|| state.last_event().map(|event| event.seq))
}

fn selected_seq_for_panel(state: &AppState, panel: Panel) -> Option<u64> {
    match panel {
        Panel::Progress => state.progress_events.get(state.selected_progress).map(|e| e.seq),
        Panel::Actions => state.action_events.get(state.selected_actions).map(|e| e.seq),
        Panel::Evidence => state.evidence_events.get(state.selected_evidence).map(|e| e.seq),
        Panel::Logs => state
            .log_lines
            .get(state.selected_logs)
            .and_then(|line| if line.seq == 0 { None } else { Some(line.seq) }),
        Panel::Metrics => state
            .profiling_events
            .get(state.selected_profiling)
            .map(|event| event.seq)
            .or_else(|| state.last_event().map(|event| event.seq)),
        _ => state.last_event().map(|event| event.seq),
    }
}

fn latest_non_log_seq(state: &AppState) -> Option<u64> {
    state
        .all_events
        .iter()
        .rev()
        .find(|event| event.event_type != "log" && !event.event.starts_with("log."))
        .map(|event| event.seq)
}

pub struct Tui {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    mouse_capture_enabled: bool,
    restored: bool,
}

impl Tui {
    pub fn init() -> io::Result<Self> {
        enable_raw_mode()?;

        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
            let _ = disable_raw_mode();
            return Err(err);
        }

        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self {
                terminal,
                mouse_capture_enabled: true,
                restored: false,
            }),
            Err(err) => {
                let _ = disable_raw_mode();
                let mut stdout = io::stdout();
                let _ = execute!(stdout, DisableMouseCapture, LeaveAlternateScreen);
                Err(err)
            }
        }
    }

    pub fn toggle_mouse_capture(&mut self) -> io::Result<()> {
        if self.mouse_capture_enabled {
            execute!(self.terminal.backend_mut(), DisableMouseCapture)?;
            self.mouse_capture_enabled = false;
        } else {
            execute!(self.terminal.backend_mut(), EnableMouseCapture)?;
            self.mouse_capture_enabled = true;
        }
        Ok(())
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        disable_raw_mode()?;
        if self.mouse_capture_enabled {
            execute!(self.terminal.backend_mut(), DisableMouseCapture)?;
        }
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        Ok(())
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
