use std::{
    io::{self, BufRead},
    sync::mpsc::{self, TryRecvError},
    thread,
    time::Duration,
};

use ratatui::crossterm::{
    event::{self, Event as CEvent, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
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
            if let CEvent::Key(key) = event::read()? {
                if handle_key(&mut state, key) {
                    break;
                }
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

fn handle_key(state: &mut AppState, key: KeyEvent) -> bool {
    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), _) => return true,
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            return true
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
    restored: bool,
}

impl Tui {
    pub fn init() -> io::Result<Self> {
        enable_raw_mode()?;

        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(err);
        }

        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self {
                terminal,
                restored: false,
            }),
            Err(err) => {
                let _ = disable_raw_mode();
                let mut stdout = io::stdout();
                let _ = execute!(stdout, LeaveAlternateScreen);
                Err(err)
            }
        }
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        disable_raw_mode()?;
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
