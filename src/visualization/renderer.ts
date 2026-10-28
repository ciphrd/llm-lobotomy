import { CHANNELS, LAYERS } from '../inference/topology'

const shader = /* wgsl */ `
struct Camera { yaw: f32, pitch: f32, distance: f32, aspect: f32, layer: f32, gain: f32, width: f32, height: f32 }
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> disabled: array<u32>;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) strength: f32, @location(2) @interpolate(flat) masked: u32 }
@vertex fn vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> Vertex {
  let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  let layer = instance / ${CHANNELS}u;
  let channel = instance % ${CHANNELS}u;
  let isolated = camera.layer >= 0.0;
  var p = vec3f((f32(channel % 128u) / 127.0 - 0.5) * 5.0, (f32(channel / 128u) / 75.0 - 0.5) * 3.0, (f32(layer) - 17.5) * 0.24);
  if (isolated) { p.z = 0.0; }
  let x = cos(camera.yaw)*p.x + sin(camera.yaw)*p.z;
  let z = -sin(camera.yaw)*p.x + cos(camera.yaw)*p.z;
  let y = cos(camera.pitch)*p.y - sin(camera.pitch)*z;
  let depth = sin(camera.pitch)*p.y + cos(camera.pitch)*z + camera.distance;
  let strength = clamp(log(1.0 + max(values[instance], 0.0)*camera.gain) / 3.0, 0.0, 1.0);
  let uv = corners[index];
  let radius = select(0.65, 1.3, isolated) + select(strength, 0.6, disabled[instance] != 0u) * 1.3;
  var out: Vertex;
  out.position = vec4f(vec2f(x*1.8/camera.aspect, y*1.8) + uv * radius * vec2f(2.0/camera.width, 2.0/camera.height)*depth, depth-0.1, depth);
  if (isolated && f32(layer) != camera.layer) { out.position = vec4f(2.0, 2.0, 1.0, 1.0); }
  out.uv = uv;
  out.strength = strength;
  out.masked = disabled[instance];
  return out;
}
@fragment fn fragment(in: Vertex) -> @location(0) vec4f {
  if (dot(in.uv, in.uv) > 1.0) { discard; }
  if (in.masked != 0u) { return vec4f(1.0, 0.12, 0.16, 1.0); }
  let color = mix(vec3f(0.08, 0.22, 0.32), mix(vec3f(0.1, 0.85, 0.8), vec3f(1.0, 0.78, 0.32), in.strength), in.strength);
  return vec4f(color, 1.0);
}`

export async function createRenderer(
  canvas: HTMLCanvasElement,
  onError: (message: string) => void,
  signal: AbortSignal,
) {
  if (!navigator.gpu)
    throw new Error(
      'WebGPU is unavailable. Open this app in a WebGPU browser on HTTPS or localhost.',
    )
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter)
    throw new Error('No WebGPU adapter is available for the visualization.')
  const device = await adapter.requestDevice()
  let disposed = false
  void device.lost.then((info) => {
    if (!disposed)
      onError(
        `Visualization GPU disconnected: ${info.message || 'reload to reconnect'}`,
      )
  })
  device.addEventListener('uncapturederror', (event) =>
    onError(event.error.message),
  )
  const context = canvas.getContext('webgpu') as GPUCanvasContext | null
  if (!context) {
    device.destroy()
    throw new Error('Unable to create a WebGPU canvas.')
  }
  const format = navigator.gpu.getPreferredCanvasFormat()
  const values = device.createBuffer({
    size: LAYERS * CHANNELS * 4,
    usage: 0x80 | 0x08 /* STORAGE | COPY_DST */,
  })
  const mask = device.createBuffer({
    size: LAYERS * CHANNELS * 4,
    usage: 0x80 | 0x08,
  })
  const uniform = device.createBuffer({
    size: 32,
    usage: 0x40 | 0x08 /* UNIFORM | COPY_DST */,
  })
  const module = device.createShaderModule({ code: shader })
  let pipeline: GPURenderPipeline
  try {
    pipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module, entryPoint: 'vertex' },
      fragment: { module, entryPoint: 'fragment', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    })
  } catch (error) {
    mask.destroy()
    values.destroy()
    uniform.destroy()
    device.destroy()
    throw error
  }
  if (signal.aborted) {
    mask.destroy()
    values.destroy()
    uniform.destroy()
    device.destroy()
    throw new Error('Visualization initialization cancelled')
  }
  context.configure({ device, format, alphaMode: 'opaque' })
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: values } },
      { binding: 1, resource: { buffer: uniform } },
      { binding: 2, resource: { buffer: mask } },
    ],
  })
  let depth: GPUTexture | undefined
  return {
    uploadMask(data: Uint32Array) {
      device.queue.writeBuffer(mask, 0, data as Uint32Array<ArrayBuffer>)
    },
    upload(data: Float32Array) {
      device.queue.writeBuffer(values, 0, data as Float32Array<ArrayBuffer>)
    },
    draw(
      yaw: number,
      pitch: number,
      distance: number,
      layer: number,
      gain: number,
    ) {
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.max(
        1,
        Math.min(
          device.limits.maxTextureDimension2D,
          Math.round(canvas.clientWidth * ratio),
        ),
      )
      const height = Math.max(
        1,
        Math.min(
          device.limits.maxTextureDimension2D,
          Math.round(canvas.clientHeight * ratio),
        ),
      )
      if (!depth || canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        depth?.destroy()
        depth = device.createTexture({
          size: [width, height],
          format: 'depth24plus',
          usage: 0x10 /* RENDER_ATTACHMENT */,
        })
      }
      device.queue.writeBuffer(
        uniform,
        0,
        new Float32Array([
          yaw,
          pitch,
          distance,
          width / height,
          layer,
          gain,
          width,
          height,
        ]),
      )
      const encoder = device.createCommandEncoder()
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.018, g: 0.035, b: 0.055, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: depth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(6, LAYERS * CHANNELS)
      pass.end()
      device.queue.submit([encoder.finish()])
    },
    dispose() {
      disposed = true
      depth?.destroy()
      mask.destroy()
      values.destroy()
      uniform.destroy()
      context.unconfigure()
      device.destroy()
    },
  }
}
