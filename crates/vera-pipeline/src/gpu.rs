use ndarray::Array2;

use crate::convolve::gaussian_1d;

/// GPU context holding a wgpu device, queue, and pre-compiled pipelines.
/// Shared (via `&GpuContext`) across Rayon threads — wgpu Device/Queue are Send+Sync.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Gaussian convolution (H + V passes)
    pipeline_h: wgpu::ComputePipeline,
    pipeline_v: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    // Kron aperture flux (one workgroup per detection)
    pipeline_kron: wgpu::ComputePipeline,
    bgl_kron: wgpu::BindGroupLayout,
}

/// Per-detection parameters uploaded to the GPU Kron shader.
/// Layout must match the WGSL `KronParams` struct exactly (repr C, 44 bytes).
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct KronParams {
    pub x_c: f32, pub y_c: f32,
    pub cos_theta: f32, pub sin_theta: f32,
    pub inv_a: f32, pub inv_b: f32,
    pub ap_sq: f32,
    pub r0: u32, pub c0: u32, pub r1: u32, pub c1: u32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct KronGlobals { ncols: u32, n_dets: u32 }

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
struct Params {
    nrows: u32,
    ncols: u32,
    radius: u32,
    kernel_len: u32,
}

impl GpuContext {
    /// Attempt to initialise a high-performance GPU context.
    /// Returns `None` if no compatible adapter is found.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        let info = adapter.get_info();
        eprintln!("  GPU : {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vera"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_bgl"),
            entries: &[
                // binding 0 — src (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1 — dst (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2 — kern (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3 — params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gaussian_h"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main_h"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_v = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gaussian_v"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main_v"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Kron aperture pipeline ────────────────────────────────────────────
        let shader_kron = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("measure"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/measure.wgsl").into()),
        });

        let bgl_kron = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kron_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let kron_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kron_layout"),
            bind_group_layouts: &[Some(&bgl_kron)],
            immediate_size: 0,
        });

        let pipeline_kron = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kron"),
            layout: Some(&kron_layout),
            module: &shader_kron,
            entry_point: Some("main_kron"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(Self { device, queue, pipeline_h, pipeline_v, bgl, pipeline_kron, bgl_kron })
    }

    /// GPU separable Gaussian smooth: two compute passes (H then V).
    /// NaN/Inf values are replaced by 0.0 before upload.
    pub fn gaussian_smooth(&self, image: &Array2<f32>, sigma: f32) -> Array2<f32> {
        let (nrows, ncols) = image.dim();
        let n = nrows * ncols;
        let buf_bytes = (n * std::mem::size_of::<f32>()) as u64;

        let kernel = gaussian_1d(sigma, 3.0);
        let radius = kernel.len() / 2;

        let img_data: Vec<f32> = image.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();

        let params = Params {
            nrows: nrows as u32,
            ncols: ncols as u32,
            radius: radius as u32,
            kernel_len: kernel.len() as u32,
        };

        // buf_a: starts with original image (uploaded via write_buffer, no extra submit).
        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_a"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // buf_b: receives the H pass output; read by V pass.
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_b"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kern_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kern"),
            size: (kernel.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Queue all uploads — they execute before the compute passes in the same submit.
        self.queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&img_data));
        self.queue.write_buffer(&kern_buf, 0, bytemuck::cast_slice(&kernel));
        self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // H pass: buf_a → buf_b
        let bg_h = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_h"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        // V pass: buf_b → buf_a
        let bg_v = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_v"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: kern_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let wg_x = ncols.div_ceil(16) as u32;
        let wg_y = nrows.div_ceil(16) as u32;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gaussian_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("h_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_h);
            pass.set_bind_group(0, &bg_h, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_v);
            pass.set_bind_group(0, &bg_v, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Result is in buf_a after the V pass.
        encoder.copy_buffer_to_buffer(&buf_a, 0, &staging, 0, buf_bytes);

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();

        Array2::from_shape_vec((nrows, ncols), result).unwrap()
    }

    /// GPU Kron aperture flux for all detections in parallel.
    /// Returns `flux_auto` for each entry in `params` (same order).
    pub fn kron_flux_batch(&self, residual: &Array2<f32>, params: &[KronParams]) -> Vec<f32> {
        if params.is_empty() {
            return Vec::new();
        }

        let (nrows, ncols) = residual.dim();
        let n_dets = params.len() as u32;
        let res_bytes = (nrows * ncols * std::mem::size_of::<f32>()) as u64;
        let params_bytes = (params.len() * std::mem::size_of::<KronParams>()) as u64;
        let out_bytes = (params.len() * std::mem::size_of::<f32>()) as u64;

        let globals = KronGlobals { ncols: ncols as u32, n_dets };

        // ── Buffers ───────────────────────────────────────────────────────────
        let buf_res = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kron_residual"),
            size: res_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kron_params"),
            size: params_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kron_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_globals = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kron_globals"),
            size: std::mem::size_of::<KronGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kron_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Upload ────────────────────────────────────────────────────────────
        let res_data: Vec<f32> = residual.iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();
        self.queue.write_buffer(&buf_res, 0, bytemuck::cast_slice(&res_data));
        self.queue.write_buffer(&buf_params, 0, bytemuck::cast_slice(params));
        self.queue.write_buffer(&buf_globals, 0, bytemuck::bytes_of(&globals));

        // ── Dispatch ──────────────────────────────────────────────────────────
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kron_bg"),
            layout: &self.bgl_kron,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_res.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_globals.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("kron_encoder") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("kron_pass"), timestamp_writes: None }
            );
            pass.set_pipeline(&self.pipeline_kron);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_dets, 1, 1); // one workgroup per detection
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        // ── Readback ─────────────────────────────────────────────────────────
        let slice = buf_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        buf_staging.unmap();
        result
    }
}
