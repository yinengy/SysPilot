fn main() {
    if let Err(err) = syspilot_tui::tui::run() {
        eprintln!("syspilot-tui: {err}");
        std::process::exit(1);
    }
}
