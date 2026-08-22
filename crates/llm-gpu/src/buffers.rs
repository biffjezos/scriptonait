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

/// Waits for `map_async` on a staging buffer and turns the result into a
/// plain `Result`.
///
/// A failed map is not a bug in the caller and must never abort the wasm
/// module: by far the most likely cause is that the *device* went away
/// mid-step (a driver reset / TDR after a long-running submission, a
/// browser-initiated device loss, or the tab losing its GPU on a
/// suspend). Panicking here took the whole wasm instance down with it,
/// which left every later call — including the training loop's own next
/// step — hanging until an unrelated 120s timeout fired, reporting a
/// timeout for what was really a lost device.
async fn await_mapped(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
) -> Result<(), String> {
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    match rx.await {
        Err(_) => Err("GPU readback was cancelled before it completed (the device was most \
                       likely lost mid-step)"
            .to_string()),
        Ok(Err(err)) => Err(format!(
            "GPU readback failed ({err:?}). This usually means the WebGPU device was lost \
             — commonly a driver watchdog reset after a long-running step. Training on CPU, \
             or a smaller model/batch size, avoids it."
        )),
        Ok(Ok(())) => Ok(()),
    }
}

/// Copies `buffer`'s first `len` f32s back to the host. Async because
/// `wgpu`'s buffer mapping is inherently async (mandatory on the web,
/// where the browser's WebGPU implementation is a JS Promise underneath);
/// natively this resolves as soon as `device.poll(PollType::Wait)` returns.
pub async fn read_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, String> {
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
    await_mapped(device, slice).await?;

    let data = slice
        .get_mapped_range()
        .map_err(|err| format!("GPU buffer reported mapped but could not be read: {err:?}"))?;
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

/// Reads several buffers back to the host as one flat, concatenated
/// `Vec<f32>` (in `buffers` order) with exactly one host-device
/// synchronization point, instead of one per buffer. Each `read_f32` call
/// is a real, expensive round trip (map a staging buffer, wait for the GPU
/// to finish everything queued so far, copy the data back) - calling it
/// once per tensor when reading, say, every weight tensor in a model (as
/// `GpuModel::read_all_weights` used to) means dozens of those round trips
/// for what's logically one read. This instead does every
/// `copy_buffer_to_buffer` into one combined staging buffer within a
/// single command buffer, submits once, and maps/awaits once - see
/// `GpuModel::train_step`'s docs for the same fix applied to loss readback.
pub async fn read_f32_concat(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &[(&wgpu::Buffer, usize)],
) -> Result<Vec<f32>, String> {
    let total_len: usize = buffers.iter().map(|(_, len)| len).sum();
    let byte_total = (total_len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-concat-staging"),
        size: byte_total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback-concat") });
    let mut offset = 0u64;
    for (buffer, len) in buffers {
        let byte_len = (*len * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, offset, byte_len);
        offset += byte_len;
    }
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    await_mapped(device, slice).await?;

    let data = slice
        .get_mapped_range()
        .map_err(|err| format!("GPU buffer reported mapped but could not be read: {err:?}"))?;
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}
