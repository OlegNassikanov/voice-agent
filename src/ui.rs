use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{enable_raw_mode, disable_raw_mode},
};
use std::io::{self, Write};

pub fn run_ui<F>(mut on_toggle: F) -> anyhow::Result<()>
where
    F: FnMut() -> (),
{
    enable_raw_mode()?;
    
    // Explicitly using print! + \r\n and flush
    print!("\r\n=== Voice Agent v0.2 (Manual Mode) ===\r\n");
    print!("\r\n[ SPACE ] Start / Stop recording\r\n");
    print!("[ ESC   ] Quit\r\n\r\n");
    io::stdout().flush()?;

    loop {
        if let Event::Key(k) = event::read()? {
            match k.code {
                KeyCode::Char(' ') => on_toggle(),
                KeyCode::Esc => break,
                KeyCode::Char('c') if k.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => break,
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    println!("\nGoodbye.");
    Ok(())
}
