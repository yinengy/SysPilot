use std::collections::VecDeque;

use crate::event::TuneEventEnvelope;

#[derive(Debug, Clone)]
pub struct LogLine {
    pub seq: u64,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct BoundedDeque<T> {
    buf: VecDeque<T>,
    cap: usize,
}

impl<T> BoundedDeque<T> {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(cap.min(1024)),
            cap: cap.max(1),
        }
    }

    pub fn push(&mut self, item: T) {
        self.buf.push_back(item);
        while self.buf.len() > self.cap {
            self.buf.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.buf.iter()
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        self.buf.get(idx)
    }

    pub fn back(&self) -> Option<&T> {
        self.buf.back()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Overview,
    Progress,
    Metrics,
    Actions,
    Evidence,
    Logs,
    Details,
}

impl Panel {
    pub const ORDER: [Panel; 7] = [
        Panel::Overview,
        Panel::Progress,
        Panel::Metrics,
        Panel::Actions,
        Panel::Evidence,
        Panel::Logs,
        Panel::Details,
    ];

    pub fn next(self) -> Self {
        let idx = Self::ORDER.iter().position(|p| *p == self).unwrap_or(0);
        Self::ORDER[(idx + 1) % Self::ORDER.len()]
    }

    pub fn prev(self) -> Self {
        let idx = Self::ORDER.iter().position(|p| *p == self).unwrap_or(0);
        Self::ORDER[(idx + Self::ORDER.len() - 1) % Self::ORDER.len()]
    }
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub focused: Panel,
    pub last_non_details_focus: Panel,
    pub pinned_seq: Option<u64>,

    pub run_id: Option<String>,
    pub schema_version: Option<String>,

    pub all_events: BoundedDeque<TuneEventEnvelope>,
    pub progress_events: BoundedDeque<TuneEventEnvelope>,
    pub profiling_events: BoundedDeque<TuneEventEnvelope>,
    pub action_events: BoundedDeque<TuneEventEnvelope>,
    pub evidence_events: BoundedDeque<TuneEventEnvelope>,
    pub log_lines: BoundedDeque<LogLine>,

    pub selected_progress: usize,
    pub selected_profiling: usize,
    pub selected_actions: usize,
    pub selected_evidence: usize,
    pub selected_logs: usize,
    pub details_scroll: usize,

    pub parse_error_count: u64,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            focused: Panel::Overview,
            last_non_details_focus: Panel::Overview,
            pinned_seq: None,
            run_id: None,
            schema_version: None,
            all_events: BoundedDeque::new(200),
            progress_events: BoundedDeque::new(200),
            profiling_events: BoundedDeque::new(120),
            action_events: BoundedDeque::new(100),
            evidence_events: BoundedDeque::new(120),
            log_lines: BoundedDeque::new(300),
            selected_progress: 0,
            selected_profiling: 0,
            selected_actions: 0,
            selected_evidence: 0,
            selected_logs: 0,
            details_scroll: 0,
            parse_error_count: 0,
        }
    }

    pub fn last_event(&self) -> Option<&TuneEventEnvelope> {
        self.all_events.back()
    }

    pub fn find_by_seq(&self, seq: u64) -> Option<&TuneEventEnvelope> {
        self.all_events.iter().rev().find(|e| e.seq == seq)
    }
}
