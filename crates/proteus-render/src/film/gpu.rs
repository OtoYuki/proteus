//! Film frames on the GPU (wgpu: Vulkan, Metal or DX12, offscreen; no window, no display
//! server). The passes mirror the CPU path ([`super::CpuWorker`]): geometry into a supersampled
//! colour and linear-depth target, a box filter onto the ground colour, then outlines and SSAO.
//! It cannot be bit-identical to the CPU across vendors (rasterisation rules and rounding
//! differ), so the CPU path stays the reference and a test holds the two within a tolerance.

use super::{Film, FrameScene};
use crate::brand;
use crate::geometry::mesh::TriangleMesh;
use crate::rasterizer::buffer::ColorRGB;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const ZBUF: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const EMPTY: f32 = 1e30;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    residue: u32,
    /// RGBA8; alpha 0 takes the colour from the per-residue table.
    color: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    rot: [[f32; 4]; 4],
    center: [f32; 4],
    pan: [f32; 4],
    scale: [f32; 4],
    depths: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    params: [u32; 4],
    ground: [f32; 4],
}

fn vertices(mesh: &TriangleMesh, color: Option<ColorRGB>) -> Vec<Vertex> {
    let c = color.map_or([0, 0, 0, 0], |c| [c.r, c.g, c.b, 255]);
    mesh.vertices
        .iter()
        .map(|v| Vertex {
            position: v.position.into(),
            normal: v.normal.into(),
            residue: v.residue_index as u32,
            color: c,
        })
        .collect()
}

fn indices(mesh: &TriangleMesh, base: u32, out: &mut Vec<u32>) {
    for t in &mesh.indices {
        out.extend([t[0] + base, t[1] + base, t[2] + base]);
    }
}

struct Target {
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

fn target(device: &wgpu::Device, w: u32, h: u32, label: &str) -> Target {
    let tex = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    Target {
        color: tex(COLOR, usage),
        depth: tex(DEPTH, usage),
    }
}

/// A GPU renderer for one film's frames.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    geometry: wgpu::RenderPipeline,
    downsample: wgpu::RenderPipeline,
    post: wgpu::RenderPipeline,
    frame_buf: wgpu::Buffer,
    colors_buf: wgpu::Buffer,
    geometry_bg: wgpu::BindGroup,
    ribbon_vb: wgpu::Buffer,
    ribbon_ib: wgpu::Buffer,
    ribbon_count: u32,
    hi: Target,
    zbuf: wgpu::TextureView,
    lo: Target,
    out: wgpu::Texture,
    down_bg: wgpu::BindGroup,
    post_bg: wgpu::BindGroup,
    readback: wgpu::Buffer,
    padded_row: u32,
    width: u32,
    height: u32,
    k: u32,
    overlay_vb: Option<(wgpu::Buffer, u64)>,
    overlay_ib: Option<(wgpu::Buffer, u64)>,
}

impl GpuRenderer {
    /// A renderer on the first hardware GPU, sized for `film`. A software adapter (llvmpipe,
    /// lavapipe, WARP) is refused: the CPU rasteriser is faster than one and is the reference.
    pub fn new(film: &Film) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!(
                "only a software adapter is available ({})",
                info.name
            ));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("proteus film"),
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device request failed: {e}"))?;

        let s = film.settings();
        let (width, height) = (s.width as u32, s.height as u32);
        let k = s.supersample.max(1) as u32;
        let limit = device.limits().max_texture_dimension_2d;
        if width * k > limit || height * k > limit {
            return Err(format!(
                "{}×{} at {k}× supersampling exceeds the GPU's {limit}-pixel textures",
                width, height
            ));
        }

        let geom_src = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("film geometry"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_geometry.wgsl").into()),
        });
        let post_src = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("film post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_post.wgsl").into()),
        });

        let geom_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("film geometry"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("film post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
            ],
        });
        let pipeline_layout = |l: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(l)],
                immediate_size: 0,
            })
        };
        let geom_pl = pipeline_layout(&geom_layout);
        let post_pl = pipeline_layout(&post_layout);

        let attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Uint32, 3 => Unorm8x4
        ];
        let two_targets = [
            Some(wgpu::ColorTargetState {
                format: COLOR,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: DEPTH,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let geometry = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("film geometry"),
            layout: Some(&geom_pl),
            vertex: wgpu::VertexState {
                module: &geom_src,
                entry_point: Some("vs_geometry"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attrs,
                })],
            },
            // No culling: ribbon faces are flipped toward the viewer in the shading, as on the
            // CPU, and tubes are open.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: ZBUF,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &geom_src,
                entry_point: Some("fs_geometry"),
                compilation_options: Default::default(),
                targets: &two_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let fullscreen = |entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&post_pl),
                vertex: wgpu::VertexState {
                    module: &post_src,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post_src,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let downsample = fullscreen("fs_downsample", &two_targets);
        let post = fullscreen("fs_post", &two_targets[..1]);

        let ribbon = film.ribbon();
        let ribbon_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ribbon vertices"),
            contents: bytemuck::cast_slice(&vertices(ribbon, None)),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut ribbon_idx = Vec::with_capacity(ribbon.indices.len() * 3);
        indices(ribbon, 0, &mut ribbon_idx);
        let ribbon_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ribbon indices"),
            contents: bytemuck::cast_slice(&ribbon_idx),
            usage: wgpu::BufferUsages::INDEX,
        });

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let residues = film.residue_count().max(1);
        let colors_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residue colours"),
            size: (residues * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let geometry_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &geom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: colors_buf.as_entire_binding(),
                },
            ],
        });

        let hi = target(&device, width * k, height * k, "hi");
        let zbuf = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("z"),
                size: wgpu::Extent3d {
                    width: width * k,
                    height: height * k,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ZBUF,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let lo = target(&device, width, height, "lo");
        let out = target(&device, width, height, "out").color;

        let ground = brand::DARK.ground;
        let post_uniform = |kk: u32| PostUniform {
            params: [kk, width, height, 0],
            ground: [
                ground.r as f32 / 255.0,
                ground.g as f32 / 255.0,
                ground.b as f32 / 255.0,
                1.0,
            ],
        };
        let post_bg = |src: &Target, kk: u32| {
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("post"),
                contents: bytemuck::bytes_of(&post_uniform(kk)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &post_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &src.color.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &src.depth.create_view(&Default::default()),
                        ),
                    },
                ],
            })
        };
        let down_bg = post_bg(&hi, k);
        let post_bg = post_bg(&lo, 1);

        let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            adapter: format!("{} ({:?})", info.name, info.backend),
            geometry,
            downsample,
            post,
            frame_buf,
            colors_buf,
            geometry_bg,
            ribbon_vb,
            ribbon_ib,
            ribbon_count: ribbon_idx.len() as u32,
            hi,
            zbuf,
            lo,
            out,
            down_bg,
            post_bg,
            readback,
            padded_row,
            width,
            height,
            k,
            overlay_vb: None,
            overlay_ib: None,
        })
    }

    /// The adapter in use, for the log line.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    fn ensure(
        slot: &mut Option<(wgpu::Buffer, u64)>,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) {
        if slot.as_ref().is_none_or(|(_, cap)| *cap < size) {
            let cap = size.next_power_of_two().max(4096);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay"),
                size: cap,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *slot = Some((buf, cap));
        }
    }

    /// Draw `scene`'s structure into `rgb` (`width × height × 3`).
    pub fn render(&mut self, scene: &FrameScene, rgb: &mut [u8]) -> Result<(), String> {
        let (w, h, k) = (self.width, self.height, self.k);
        let cam = &scene.camera;
        let r = cam.rotation_matrix();
        let scale = 1.0 / cam.angstroms_per_pixel((w * k) as usize, (h * k) as usize) as f32;
        let (fog_lo, fog_hi) = scene.depth_range;
        let far = fog_hi.abs().max(fog_lo.abs()) * 4.0 + 1000.0;
        let near = scene.clip_near.unwrap_or(-far);
        let uniform = FrameUniform {
            // WGSL matrices are column-major.
            rot: [
                [r[(0, 0)], r[(1, 0)], r[(2, 0)], 0.0],
                [r[(0, 1)], r[(1, 1)], r[(2, 1)], 0.0],
                [r[(0, 2)], r[(1, 2)], r[(2, 2)], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            center: [cam.center.x, cam.center.y, cam.center.z, 0.0],
            pan: [cam.pan.x, cam.pan.y, cam.pan.z, 0.0],
            scale: [scale, (w * k) as f32, (h * k) as f32, 0.0],
            depths: [near, far, fog_lo, fog_hi],
        };
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&uniform));
        let packed: Vec<u32> = scene
            .residue_colors
            .iter()
            .map(|c| c.r as u32 | (c.g as u32) << 8 | (c.b as u32) << 16)
            .collect();
        if !packed.is_empty() {
            self.queue
                .write_buffer(&self.colors_buf, 0, bytemuck::cast_slice(&packed));
        }

        let mut ov: Vec<Vertex> = Vec::new();
        let mut oi: Vec<u32> = Vec::new();
        for (mesh, color) in &scene.overlays {
            indices(mesh, ov.len() as u32, &mut oi);
            ov.extend(vertices(mesh, Some(*color)));
        }
        if !oi.is_empty() {
            let vb_size = std::mem::size_of_val(ov.as_slice()) as u64;
            let ib_size = std::mem::size_of_val(oi.as_slice()) as u64;
            Self::ensure(
                &mut self.overlay_vb,
                &self.device,
                vb_size,
                wgpu::BufferUsages::VERTEX,
            );
            Self::ensure(
                &mut self.overlay_ib,
                &self.device,
                ib_size,
                wgpu::BufferUsages::INDEX,
            );
            if let (Some((vb, _)), Some((ib, _))) = (&self.overlay_vb, &self.overlay_ib) {
                self.queue.write_buffer(vb, 0, bytemuck::cast_slice(&ov));
                self.queue.write_buffer(ib, 0, bytemuck::cast_slice(&oi));
            }
        }

        let ground = brand::DARK.ground;
        let ground_color = wgpu::Color {
            r: ground.r as f64 / 255.0,
            g: ground.g as f64 / 255.0,
            b: ground.b as f64 / 255.0,
            a: 1.0,
        };
        let empty = wgpu::Color {
            r: EMPTY as f64,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let (hi_c, hi_d, lo_c, lo_d, out_v) = (
            view(&self.hi.color),
            view(&self.hi.depth),
            view(&self.lo.color),
            view(&self.lo.depth),
            view(&self.out),
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("geometry"),
                color_attachments: &[attach(&hi_c, ground_color), attach(&hi_d, empty)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.zbuf,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.geometry);
            pass.set_bind_group(0, &self.geometry_bg, &[]);
            pass.set_vertex_buffer(0, self.ribbon_vb.slice(..));
            pass.set_index_buffer(self.ribbon_ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.ribbon_count, 0, 0..1);
            if !oi.is_empty() {
                if let (Some((vb, _)), Some((ib, _))) = (&self.overlay_vb, &self.overlay_ib) {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..oi.len() as u32, 0, 0..1);
                }
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("downsample"),
                color_attachments: &[attach(&lo_c, ground_color), attach(&lo_d, empty)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.downsample);
            pass.set_bind_group(0, &self.down_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[attach(&out_v, ground_color)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post);
            pass.set_bind_group(0, &self.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.out,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("GPU readback failed: {e}"))?;
        {
            let data = slice
                .get_mapped_range()
                .map_err(|e| format!("GPU readback failed: {e}"))?;
            for y in 0..h as usize {
                let row = &data[y * self.padded_row as usize..][..w as usize * 4];
                let dst = &mut rgb[y * w as usize * 3..][..w as usize * 3];
                for (px, out) in row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<3>().0)
                {
                    out.copy_from_slice(&px[..3]);
                }
            }
        }
        self.readback.unmap();
        Ok(())
    }
}

fn attach(
    v: &wgpu::TextureView,
    clear: wgpu::Color,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view: v,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear),
            store: wgpu::StoreOp::Store,
        },
    })
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl Film {
    /// Render `frame` on the GPU into `rgb`, captions included.
    pub fn render_gpu(
        &self,
        frame: usize,
        gpu: &mut GpuRenderer,
        text: &super::text::TextPainter,
        rgb: &mut [u8],
    ) -> Result<(), String> {
        let scene = self.scene(frame);
        gpu.render(&scene, rgb)?;
        self.draw_overlay(frame, rgb, text);
        Ok(())
    }
}
