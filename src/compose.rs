//! Сборка полного кадра из двух текстур (для хоста, который встраивает GPUI в свой пайплайн): фон (кадр хоста)
//! и поверх него слой, нарисованный GPUI (`WgpuRenderer::draw_to`), цвет слоя умножен на альфу.
//! Один полноэкранный проход, без пересчёта вёрстки: слой GPUI перерисовывает только при изменениях,
//! а сборка идёт на каждый кадр фона.
//!
//! Строки на GLES: фон и выход — как у GL-текстуры хоста (первая строка в памяти — низ кадра),
//! слой — как у wgpu (первая строка — верх), поэтому слой читается перевёрнутым.

/// Конвейер сборки. Источники (`source`) создаются один раз на текстуру и переиспользуются.
pub struct Composer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

const SHADER: &str = r#"
@group(0) @binding(0) var bg_tex: texture_2d<f32>;
@group(0) @binding(1) var bg_smp: sampler;
@group(1) @binding(0) var layer_tex: texture_2d<f32>;
@group(1) @binding(1) var layer_smp: sampler;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    // Треугольник на весь экран; uv.y = 0 — верх цели (clip y = +1).
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}
@fragment fn fs(v: Out) -> @location(0) vec4<f32> {
    let bg = textureSample(bg_tex, bg_smp, v.uv);
    let layer = textureSample(layer_tex, layer_smp, vec2<f32>(v.uv.x, 1.0 - v.uv.y));
    return vec4<f32>(layer.rgb + bg.rgb * (1.0 - layer.a), 1.0);
}
"#;

impl Composer {
    /// `format` — формат выходной текстуры.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose_source"),
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
            label: Some("compose"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compose"),
            bind_group_layouts: &[Some(&layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compose"),
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
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("compose"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, layout, sampler }
    }

    /// Источник сборки (фон или слой) из текстуры — один раз на текстуру.
    pub fn source(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compose_source"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    /// Фон + слой → `out` (команды уходят в очередь; дождаться GPU — забота вызывающего).
    pub fn compose(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bg: &wgpu::BindGroup,
        layer: &wgpu::BindGroup,
        out: &wgpu::TextureView,
    ) {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compose") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, layer, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }
}

/// GL-текстура хоста (из общей группы контекстов) как текстура wgpu, RGBA8 `w`x`h`, только для чтения в шейдере.
/// Владеет ей хост: wgpu её не удалит.
///
/// # Safety
/// `name` — живая 2D-текстура RGBA8 этого размера в группе текущего GL-контекста wgpu, живёт дольше результата.
#[cfg(target_os = "android")]
pub unsafe fn host_gl_texture(device: &wgpu::Device, name: u32, w: u32, h: u32) -> Option<wgpu::Texture> {
    let raw = std::num::NonZeroU32::new(name)?;
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("host_gl_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::wgt::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };
    // SAFETY: по договору функции текстура живая и такого формата; drop_callback — wgpu не удаляет её сам.
    let hal_tex = unsafe {
        let dev = device.as_hal::<wgpu::hal::api::Gles>()?;
        dev.texture_from_raw(raw, &hal_desc, Some(Box::new(|| {})))
    };
    let desc = wgpu::TextureDescriptor {
        label: Some("host_gl_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: hal_tex создан на этом устройстве по desc выше.
    Some(unsafe { device.create_texture_from_hal::<wgpu::hal::api::Gles>(hal_tex, &desc) })
}
