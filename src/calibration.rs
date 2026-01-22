/// Voice calibration system for personalized Whisper transcription.
/// Creates a voice profile from reference phrases to improve accuracy.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use crate::audio::AudioRecorder;
use crate::audio_processor::AudioProcessor;
use crate::whisper::WhisperModel;

/// Calibration phrases in Russian - designed to cover common sounds and vocabulary
pub const CALIBRATION_PHRASES: &[&str] = &[
    "Раз два три четыре пять. Шесть семь восемь девять десять.",
    "Всем привет папа здесь. Сегодня отличная погода.",
    "Где купить лопаты два миллиона рублей. Удалить прикрепить стереть.",
    "Мы купим горячие котлеты. Не пойдёт в принципе неплохо.",
    "Говорю чётко и медленно на русском языке.",
    "Кошка мяукает собака лает. Компьютер работает быстро.",
];

/// Voice profile containing calibration data
#[derive(Serialize, Deserialize, Default)]
pub struct VoiceProfile {
    /// Calibration prompt text (last 300 chars of transcriptions)
    pub prompt: String,
    /// ISO timestamp when profile was created
    pub created_at: String,
}

impl VoiceProfile {
    /// Get the config directory path
    fn config_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("voice-agent"))
    }

    /// Get the profile file path
    fn profile_path() -> Option<PathBuf> {
        Self::config_dir().map(|p| p.join("profile.json"))
    }

    /// Load profile from disk, returns None if not found
    pub fn load() -> Option<Self> {
        let path = Self::profile_path()?;
        let data = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Save profile to disk
    pub fn save(&self) -> anyhow::Result<()> {
        let dir = Self::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot find config directory"))?;
        
        fs::create_dir_all(&dir)?;
        
        let path = dir.join("profile.json");
        let data = serde_json::to_string_pretty(self)?;
        fs::write(&path, data)?;
        
        Ok(())
    }

    /// Check if profile exists
    pub fn exists() -> bool {
        Self::profile_path()
            .map(|p| p.exists())
            .unwrap_or(false)
    }
}

/// Run the calibration process interactively
pub fn run_calibration(
    whisper: &WhisperModel,
    recorder: &AudioRecorder,
) -> anyhow::Result<VoiceProfile> {
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
    use crossterm::event::{self, Event, KeyCode};
    
    // Disable raw mode for calibration (we need normal input)
    let _ = disable_raw_mode();
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           🎙️  КАЛИБРОВКА ГОЛОСА / VOICE CALIBRATION          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Прочитайте каждую фразу чётко, на расстоянии 15-20 см от    ║");
    println!("║  микрофона. Нажмите ПРОБЕЛ для начала записи, ещё раз для    ║");
    println!("║  остановки. ESC для пропуска фразы.                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let processor = AudioProcessor::default();
    let mut collected_text = String::new();

    enable_raw_mode()?;

    for (i, phrase) in CALIBRATION_PHRASES.iter().enumerate() {
        print!("\r\n📝 Фраза {}/{}: \"{}\"\r\n", i + 1, CALIBRATION_PHRASES.len(), phrase);
        print!("   [ ПРОБЕЛ ] Начать запись  [ ESC ] Пропустить\r\n");
        io::stdout().flush()?;

        // Wait for space to start
        loop {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char(' ') => break,
                    KeyCode::Esc => {
                        print!("   ⏭️  Пропущено\r\n");
                        io::stdout().flush()?;
                        continue;
                    }
                    _ => {}
                }
            }
        }

        // Start recording
        print!("   🔴 Записываю... (ПРОБЕЛ для остановки)\r\n");
        io::stdout().flush()?;
        
        let stream = recorder.start();
        
        // Wait for space to stop
        loop {
            if let Event::Key(k) = event::read()? {
                if k.code == KeyCode::Char(' ') {
                    break;
                }
            }
        }
        
        drop(stream);
        let audio = recorder.stop();

        if audio.is_empty() {
            print!("   ⚠️  Нет аудио, попробуйте ещё раз\r\n");
            io::stdout().flush()?;
            continue;
        }

        // Process and transcribe
        let chunks = processor.process(&audio);
        if chunks.is_empty() {
            print!("   ⚠️  Речь не обнаружена\r\n");
            io::stdout().flush()?;
            continue;
        }

        match whisper.transcribe_chunks(&chunks) {
            Ok(text) => {
                let trimmed = text.trim();
                if trimmed.len() > 5 {
                    print!("   ✅ Записано: \"{}\"\r\n", trimmed);
                    collected_text.push(' ');
                    collected_text.push_str(trimmed);
                } else {
                    print!("   ⚠️  Слишком коротко, попробуйте ещё раз\r\n");
                }
            }
            Err(e) => {
                print!("   ❌ Ошибка: {}\r\n", e);
            }
        }
        io::stdout().flush()?;
    }

    disable_raw_mode()?;

    // Build profile (last 300 chars)
    let prompt = if collected_text.len() > 300 {
        collected_text[collected_text.len() - 300..].to_string()
    } else {
        collected_text.trim().to_string()
    };

    let profile = VoiceProfile {
        prompt,
        created_at: chrono_lite_now(),
    };

    // Save profile
    profile.save()?;

    println!("\n✅ Калибровка завершена! Профиль сохранён.");
    println!("   Путь: {:?}", VoiceProfile::profile_path());

    Ok(profile)
}

/// Simple timestamp without external chrono dependency
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}
