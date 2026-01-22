mod audio;
mod audio_processor;
mod calibration;
mod whisper;
mod ui;

use audio::AudioRecorder;
use audio_processor::AudioProcessor;
use calibration::{run_calibration, VoiceProfile};
use whisper::WhisperModel;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::io::{self, Write};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let force_calibrate = args.iter().any(|a| a == "--calibrate" || a == "-c");

    // Initialize model once at startup
    println!("Loading model...");
    let mut whisper_model = WhisperModel::new("models/ggml-base.bin")?;
    println!("Model loaded!");

    let recorder = AudioRecorder::new();

    // Handle calibration
    if force_calibrate || !VoiceProfile::exists() {
        if !VoiceProfile::exists() {
            println!("\n⚠️  No voice profile found. Starting calibration...");
        }
        let profile = run_calibration(&whisper_model, &recorder)?;
        whisper_model.set_calibration_prompt(&profile.prompt);
    } else if let Some(profile) = VoiceProfile::load() {
        println!("✅ Voice profile loaded");
        whisper_model.set_calibration_prompt(&profile.prompt);
    }

    let recording = Arc::new(AtomicBool::new(false));
    
    // We keep the stream in a mutable option to drop it (stop it) when toggling off
    let mut stream = None;

    ui::run_ui({
        let recording = recording.clone();
        // Move whisper_model into the closure
        move || {
            // Toggle logic
            if !recording.load(Ordering::SeqCst) {
                // START
                print!("\r🎙  Recording... (Press SPACE to stop)   ");
                io::stdout().flush().unwrap();
                
                stream = Some(recorder.start());
                recording.store(true, Ordering::SeqCst);
            } else {
                // STOP
                print!("\r⏹  Processing...                        ");
                io::stdout().flush().unwrap();
                
                // Drop the stream to stop capturing
                drop(stream.take());
                
                // Get audio
                let audio = recorder.stop();
                recording.store(false, Ordering::SeqCst);

                if audio.is_empty() {
                    print!("\r⚠️  No audio recorded.\r\n");
                    io::stdout().flush().unwrap();
                } else {
                    // Process audio: trim silence, normalize, chunk
                    let processor = AudioProcessor::default();
                    let chunks = processor.process(&audio);
                    
                    if chunks.is_empty() {
                        print!("\r⚠️  No speech detected.\r\n");
                        io::stdout().flush().unwrap();
                    } else {
                        print!("\r⏳ Transcribing {} chunk(s)...\r\n", chunks.len());
                        io::stdout().flush().unwrap();
                        
                        // Transcribe using chunked method with context
                        match whisper_model.transcribe_chunks(&chunks) {
                            Ok(text) => {
                                print!("\r📝 RESULT: {}\r\n", text.trim());
                                print!("\r[ SPACE ] Ready\r\n");
                                io::stdout().flush().unwrap();
                            }
                            Err(e) => {
                                eprint!("\r❌ Error: {}\r\n", e);
                                io::stdout().flush().unwrap();
                            }
                        }
                    }
                }
            }
        }
    })?;

    Ok(())
}