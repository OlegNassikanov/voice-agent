use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::{Arc, Mutex};

pub struct AudioRecorder {
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&self) -> cpal::Stream {
        let host = cpal::default_host();
        let device = host.default_input_device().unwrap();

        let config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16_000),
            buffer_size: cpal::BufferSize::Fixed(1600),
        };

        let buffer = self.buffer.clone();

        let err_fn = |e| eprintln!("cpal error: {}", e);

        let format = device.default_input_config().unwrap().sample_format();

        let stream = match format {
            SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    if let Ok(mut b) = buffer.lock() {
                        b.extend_from_slice(data);
                    }
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    if let Ok(mut b) = buffer.lock() {
                        for v in data {
                            b.push(*v as f32 / i16::MAX as f32);
                        }
                    }
                },
                err_fn,
                None,
            ),
            _ => panic!("Unsupported format"),
        }.unwrap();

        stream.play().unwrap();
        stream
    }

    pub fn stop(&self) -> Vec<f32> {
        if let Ok(mut b) = self.buffer.lock() {
            b.drain(..).collect()
        } else {
            Vec::new()
        }
    }
}
