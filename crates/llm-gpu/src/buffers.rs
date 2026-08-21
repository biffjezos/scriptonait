//! Small helpers for creating/writing/reading GPU buffers, so
//! `model.rs`'s forward pass reads as a sequence of ops rather than
//! repeated wgpu boilerplate.

use wgpu::util::DeviceExt;

pub fn upload_f32(device: &wgpu::Device, label: &str, data: &[f32], extra_usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
    })
}

pub fn upload_u32(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// A zero-initialized storage buffer big enough for `len` f32s, usable
/// both as a shader read/write target and (if `readable`) as the source
/// of a later `read_f32` copy.
pub fn storage_f32(device: &wgpu::Device, label: &str, len: usize, readable: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if readable {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

pub fn uniform<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, value: T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn write_uniform<T: bytemuck::Pod>(queue: &wgpu::Queue, buffer: &wgpu::Buffer, value: T) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&value));
}

pub fn write_u32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u32]) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}

/// Copies `buffer`'s first `len` f32s back to the host. Async because
/// `wgpu`'s buffer mapping is inherently async (mandatory on the web,
/// where the browser's WebGPU implementation is a JS Promise underneath);
/// natively this resolves as soon as `device.poll(PollType::Wait)` returns.
pub async fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.await
        .expect("map_async callback dropped without firing")
        .expect("failed to map GPU buffer for readback");

    let data = slice.get_mapped_range().expect("buffer was just confirmed mapped above");
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
