use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use std::ffi::c_void;

pub struct WhisperModel {
    ctx: WhisperContext,
    /// Calibration prompt for improved accuracy (set from voice profile)
    calibration_prompt: Option<String>,
}

impl WhisperModel {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        // Suppress logs
        unsafe {
            whisper_rs::set_log_callback(Some(null_log_callback), std::ptr::null_mut());
        }

        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        
        Ok(Self { 
            ctx,
            calibration_prompt: None,
        })
    }

    /// Set the calibration prompt from voice profile
    pub fn set_calibration_prompt(&mut self, prompt: &str) {
        if !prompt.is_empty() {
            self.calibration_prompt = Some(prompt.to_string());
        }
    }

    pub fn transcribe(&self, audio: &[f32]) -> anyhow::Result<String> {
        let mut state = self.ctx.create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create state: {}", e))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_language(Some("ru"));

        // Apply calibration prompt if set
        if let Some(ref prompt) = self.calibration_prompt {
            params.set_initial_prompt(prompt);
        }

        state.full(params, audio)
            .map_err(|e| anyhow::anyhow!("Failed to run model: {}", e))?;

        let mut text = String::new();
        let num_segments = state.full_n_segments().unwrap_or(0);
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        Ok(text)
    }

    /// Transcribe multiple audio chunks with context continuity
    /// Uses the end of previous transcription as prompt for next chunk
    pub fn transcribe_chunks(&self, chunks: &[Vec<f32>]) -> anyhow::Result<String> {
        if chunks.is_empty() {
            return Ok(String::new());
        }

        // If only one chunk, use regular transcription
        if chunks.len() == 1 {
            return self.transcribe(&chunks[0]);
        }

        let mut full_text = String::new();

        for chunk in chunks {
            let mut state = self.ctx.create_state()
                .map_err(|e| anyhow::anyhow!("Failed to create state: {}", e))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_print_progress(false);
            params.set_print_special(false);
            params.set_language(Some("ru"));

            // Build prompt: calibration + previous context
            let prompt = match (&self.calibration_prompt, full_text.is_empty()) {
                (Some(cal), true) => cal.clone(),
                (Some(cal), false) => {
                    let ctx_start = full_text.len().saturating_sub(100);
                    format!("{} {}", cal, &full_text[ctx_start..])
                }
                (None, false) => {
                    let ctx_start = full_text.len().saturating_sub(100);
                    full_text[ctx_start..].to_string()
                }
                (None, true) => String::new(),
            };
            
            if !prompt.is_empty() {
                params.set_initial_prompt(&prompt);
            }

            state.full(params, chunk)
                .map_err(|e| anyhow::anyhow!("Failed to run model: {}", e))?;

            let num_segments = state.full_n_segments().unwrap_or(0);
            for i in 0..num_segments {
                if let Ok(segment) = state.full_get_segment_text(i) {
                    full_text.push_str(&segment);
                }
            }
        }

        Ok(full_text)
    }
}

extern "C" fn null_log_callback(_level: u32, _message: *const i8, _user_data: *mut c_void) {
    // Do nothing
}