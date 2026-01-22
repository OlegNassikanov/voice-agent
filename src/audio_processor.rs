/// Audio processor for improving Whisper transcription quality.
/// Implements chunking, silence trimming, and normalization.

const SAMPLE_RATE: usize = 16_000;

/// Configuration for audio processing
pub struct AudioProcessor {
    /// Duration of each chunk in seconds (default: 25s - optimal for Whisper)
    pub chunk_duration_secs: f32,
    /// Overlap between chunks in seconds (default: 2s)
    pub overlap_secs: f32,
    /// Silence threshold in dB (default: -30 dB)
    pub silence_threshold_db: f32,
    /// Minimum chunk duration to keep (in seconds)
    pub min_chunk_secs: f32,
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self {
            chunk_duration_secs: 25.0,
            overlap_secs: 2.0,
            silence_threshold_db: -30.0,
            min_chunk_secs: 1.0,
        }
    }
}

impl AudioProcessor {
    /// Calculate RMS (Root Mean Square) energy of audio segment
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    /// Convert RMS to dB
    fn rms_to_db(rms: f32) -> f32 {
        if rms <= 0.0 {
            return -100.0;
        }
        20.0 * rms.log10()
    }

    /// Convert dB threshold to linear amplitude
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Trim silence from the beginning and end of audio
    fn trim_silence(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        let threshold = Self::db_to_linear(self.silence_threshold_db);
        let frame_size = SAMPLE_RATE / 100; // 10ms frames

        // Find start (first frame above threshold)
        let mut start = 0;
        for i in (0..audio.len()).step_by(frame_size) {
            let end = (i + frame_size).min(audio.len());
            let rms = Self::calculate_rms(&audio[i..end]);
            if rms > threshold {
                start = i;
                break;
            }
        }

        // Find end (last frame above threshold)
        let mut end = audio.len();
        for i in (0..audio.len()).step_by(frame_size).rev() {
            let frame_end = (i + frame_size).min(audio.len());
            let rms = Self::calculate_rms(&audio[i..frame_end]);
            if rms > threshold {
                end = frame_end;
                break;
            }
        }

        if start >= end {
            return Vec::new();
        }

        audio[start..end].to_vec()
    }

    /// Normalize audio to [-1.0, 1.0] range
    fn normalize(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        let max_val = audio.iter()
            .map(|s| s.abs())
            .fold(0.0_f32, f32::max);

        if max_val < 1e-6 {
            return audio.to_vec();
        }

        // Normalize to 0.95 to avoid clipping
        let scale = 0.95 / max_val;
        audio.iter().map(|s| s * scale).collect()
    }

    /// Split audio into chunks with overlap
    fn chunk_with_overlap(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let chunk_samples = (self.chunk_duration_secs * SAMPLE_RATE as f32) as usize;
        let overlap_samples = (self.overlap_secs * SAMPLE_RATE as f32) as usize;
        let min_samples = (self.min_chunk_secs * SAMPLE_RATE as f32) as usize;

        // If audio is shorter than chunk size, return as single chunk
        if audio.len() <= chunk_samples {
            if audio.len() >= min_samples {
                return vec![audio.to_vec()];
            } else {
                return Vec::new();
            }
        }

        let step = chunk_samples - overlap_samples;
        let mut chunks = Vec::new();
        let mut pos = 0;

        while pos < audio.len() {
            let end = (pos + chunk_samples).min(audio.len());
            let chunk = audio[pos..end].to_vec();
            
            if chunk.len() >= min_samples {
                chunks.push(chunk);
            }

            pos += step;

            // Avoid tiny last chunk
            if audio.len() - pos < min_samples && !chunks.is_empty() {
                break;
            }
        }

        chunks
    }

    /// Main processing pipeline: trim → normalize → chunk
    pub fn process(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        // Step 1: Trim leading/trailing silence
        let trimmed = self.trim_silence(audio);
        
        if trimmed.is_empty() {
            return Vec::new();
        }

        // Step 2: Normalize
        let normalized = self.normalize(&trimmed);

        // Step 3: Chunk with overlap
        self.chunk_with_overlap(&normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_calculation() {
        let silence = vec![0.0; 1000];
        assert_eq!(AudioProcessor::calculate_rms(&silence), 0.0);

        let signal = vec![1.0; 1000];
        assert!((AudioProcessor::calculate_rms(&signal) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_trim_silence() {
        let processor = AudioProcessor::default();
        
        // Create audio with silence at start/end
        let mut audio = vec![0.0; 1600]; // 100ms silence
        audio.extend(vec![0.5; 16000]);  // 1s of signal
        audio.extend(vec![0.0; 1600]);   // 100ms silence
        
        let trimmed = processor.trim_silence(&audio);
        assert!(trimmed.len() < audio.len());
        assert!(trimmed.len() >= 16000);
    }

    #[test]
    fn test_normalize() {
        let processor = AudioProcessor::default();
        let audio = vec![0.1, -0.2, 0.15];
        let normalized = processor.normalize(&audio);
        
        let max = normalized.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!((max - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_chunking() {
        let processor = AudioProcessor {
            chunk_duration_secs: 2.0,
            overlap_secs: 0.5,
            min_chunk_secs: 0.5,
            ..Default::default()
        };

        // 5 seconds of audio
        let audio = vec![0.5; 5 * SAMPLE_RATE];
        let chunks = processor.chunk_with_overlap(&audio);
        
        assert!(chunks.len() >= 2);
        // Each chunk should be 2 seconds
        assert_eq!(chunks[0].len(), 2 * SAMPLE_RATE);
    }
}
