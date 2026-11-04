//! Текстуры хоста в сцене GPUI (Android): элемент сцены `PaintSurface` с `HostTexture` рисуется не из атласа,
//! а из текстуры, которую хост отдал рендеру под номером (`WgpuRenderer::set_host_texture`) — кадр камеры и т. п.
//! Поворот, зеркало и область текстуры — в шейдере, скругление углов — расстоянием до скруглённого прямоугольника.
//! Строки текстуры — как у wgpu (первая строка — верх кадра).
#![cfg(target_os = "android")]

use bytemuck::{Pod, Zeroable};

/// Сколько текстур хоста можно нарисовать за кадр (по одной записи параметров на каждую).
const MAX_PER_FRAME: u64 = 16;
/// Шаг записей параметров в буфере (min_uniform_buffer_offset_alignment у всех устройств ≤ 256).
const STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    bounds: [f32; 4],
    mask: [f32; 4],
    radii: [f32; 4],
    uv: [f32; 4],
    viewport: [f32; 2],
    rotation: u32,
    mirror: u32,
}

const SHADER: &str = r#"
struct Params {
    bounds: vec4<f32>,
    mask: vec4<f32>,
    radii: vec4<f32>,   // tl, tr, br, bl
    uv: vec4<f32>,
    viewport: vec2<f32>,
    rotation: u32,
    mirror: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var s: sampler;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) local: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let unit = vec2<f32>(f32(i & 1u), f32((i >> 1u) & 1u));
    let px = p.bounds.xy + unit * p.bounds.zw;
    var o: Out;
    o.pos = vec4<f32>(px.x / p.viewport.x * 2.0 - 1.0, 1.0 - px.y / p.viewport.y * 2.0, 0.0, 1.0);
    o.local = unit;
    return o;
}
@fragment fn fs(v: Out) -> @location(0) vec4<f32> {
    let px = p.bounds.xy + v.local * p.bounds.zw;
    if (px.x < p.mask.x || px.y < p.mask.y || px.x > p.mask.x + p.mask.z || px.y > p.mask.y + p.mask.w) {
        discard;
    }
    // Точка элемента → точка кадра (область uv — в кадре после поворота и зеркала) → точка текстуры.
    var q = p.uv.xy + v.local * p.uv.zw;
    if (p.mirror == 1u) { q.x = 1.0 - q.x; }
    var tc = q;
    if (p.rotation == 90u) { tc = vec2<f32>(q.y, 1.0 - q.x); }
    else if (p.rotation == 180u) { tc = vec2<f32>(1.0 - q.x, 1.0 - q.y); }
    else if (p.rotation == 270u) { tc = vec2<f32>(1.0 - q.y, q.x); }
    let c = textureSample(t, s, tc);
    // Скруглённый прямоугольник: радиус угла по четверти, в которой точка.
    let half = p.bounds.zw * 0.5;
    let d = px - (p.bounds.xy + half);
    var r = p.radii.x;
    if (d.x >= 0.0 && d.y < 0.0) { r = p.radii.y; }
    else if (d.x >= 0.0 && d.y >= 0.0) { r = p.radii.z; }
    else if (d.x < 0.0 && d.y >= 0.0) { r = p.radii.w; }
    let k = abs(d) - (half - vec2<f32>(r));
    let dist = length(max(k, vec2<f32>(0.0))) + min(max(k.x, k.y), 0.0) - r;
    let a = clamp(0.5 - dist, 0.0, 1.0);
    // Цвет текстуры считается умноженным на альфу (у камеры альфа 1).
    return c * a;
}
"#;

pub(crate) struct HostTextures {
    pipeline: wgpu::RenderPipeline,
    tex_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    params_group: wgpu::BindGroup,
    textures: Vec<Option<wgpu::BindGroup>>,
}

impl HostTextures {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("host_texture_params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            }],
        });
        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("host_texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("host_texture"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("host_texture"),
            bind_group_layouts: &[Some(&params_layout), Some(&tex_layout)],
            immediate_size: 0,
        });
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("host_texture"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(blend), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("host_texture"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("host_texture_params"),
            size: STRIDE * MAX_PER_FRAME,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("host_texture_params"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                }),
            }],
        });
        Self { pipeline, tex_layout, sampler, params, params_group, textures: Vec::new() }
    }

    pub(crate) fn set(&mut self, device: &wgpu::Device, id: u32, view: Option<&wgpu::TextureView>) {
        let i = id as usize;
        if self.textures.len() <= i {
            self.textures.resize_with(i + 1, || None);
        }
        self.textures[i] = view.map(|v| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("host_texture"),
                layout: &self.tex_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            })
        });
    }

    /// Нарисовать текстуры хоста из пачки сцены. `slot` — счётчик записей параметров в этом кадре.
    pub(crate) fn draw(
        &self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        surfaces: &[gpui::PaintSurface],
        viewport: [f32; 2],
        slot: &mut u64,
    ) {
        for s in surfaces {
            let Some(Some(tex)) = self.textures.get(s.texture.id as usize) else { continue };
            if *slot >= MAX_PER_FRAME {
                log::warn!("host_texture: больше {MAX_PER_FRAME} текстур хоста за кадр");
                return;
            }
            let b = s.bounds;
            let m = s.content_mask.bounds;
            let r = s.corner_radii;
            let params = Params {
                bounds: [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0],
                mask: [m.origin.x.0, m.origin.y.0, m.size.width.0, m.size.height.0],
                radii: [r.top_left.0, r.top_right.0, r.bottom_right.0, r.bottom_left.0],
                uv: s.texture.uv,
                viewport,
                rotation: s.texture.rotation as u32,
                mirror: s.texture.mirror as u32,
            };
            let offset = *slot * STRIDE;
            queue.write_buffer(&self.params, offset, bytemuck::bytes_of(&params));
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.params_group, &[offset as u32]);
            pass.set_bind_group(1, tex, &[]);
            pass.draw(0..4, 0..1);
            *slot += 1;
        }
    }
}
