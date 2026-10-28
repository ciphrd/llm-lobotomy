import proto from 'onnx-proto'
import type { onnx as Onnx } from 'onnx-proto'
const { onnx } = proto

export const MODEL_ID = 'onnx-community/Qwen3-4B-ONNX'
export const REVISION = 'ca652ef636e8b121ff4a1607c37ab578cb866987'
import { LAYERS, CHANNELS } from './topology.ts'
export { LAYERS, CHANNELS } from './topology.ts'
export const maskName = (layer: number) => `ablation_mask_${layer}`
export const activationName = (layer: number) => `activation_mean_${layer}`

/** Insert real MLP channel masks, after SiLU(gate) * up and before down_proj. */
export function instrumentGraph(bytes: Uint8Array): Uint8Array {
  const model = onnx.ModelProto.decode(bytes)
  const graph = model.graph
  if (!graph) throw new Error('Missing ONNX graph')
  graph.node ??= []
  graph.input ??= []
  graph.output ??= []
  const found = new Set<number>()
  const nodes: Onnx.INodeProto[] = []
  for (const node of graph.node) {
    const match = node.name?.match(
      /^\/model\/layers\.(\d+)\/mlp\/down_proj\/MatMul_Q4$/,
    )
    if (match) {
      if (!node.input?.length) throw new Error('Missing MLP input')
      const layer = Number(match[1])
      const source = node.input[0]
      const width = Number(node.attribute?.find((a) => a.name === 'K')?.i)
      if (width !== CHANNELS || found.has(layer))
        throw new Error('Unexpected Qwen MLP structure')
      found.add(layer)
      graph.input.push(
        onnx.ValueInfoProto.create({
          name: maskName(layer),
          type: {
            tensorType: {
              elemType: 10,
              shape: { dim: [{ dimValue: CHANNELS }] },
            },
          },
        }),
      )
      const masked = `masked_mlp_${layer}`
      nodes.push(
        onnx.NodeProto.create({
          name: masked,
          opType: 'Mul',
          input: [source, maskName(layer)],
          output: [masked],
        }),
      )
      node.input[0] = masked
      // Pre-mask summaries support profiling; live display applies the current mask.
      const cast = `activation_float_${layer}`
      const abs = `activation_abs_${layer}`
      nodes.push(
        onnx.NodeProto.create({
          name: cast,
          opType: 'Cast',
          input: [source],
          output: [cast],
          attribute: [{ name: 'to', type: 2, i: 1 }],
        }),
      )
      nodes.push(
        onnx.NodeProto.create({
          name: abs,
          opType: 'Abs',
          input: [cast],
          output: [abs],
        }),
      )
      nodes.push(
        onnx.NodeProto.create({
          name: activationName(layer),
          opType: 'ReduceMean',
          input: [abs],
          output: [activationName(layer)],
          attribute: [
            { name: 'axes', type: 7, ints: [0, 1] },
            { name: 'keepdims', type: 2, i: 0 },
          ],
        }),
      )
      graph.output.push(
        onnx.ValueInfoProto.create({
          name: activationName(layer),
          type: {
            tensorType: {
              elemType: 1,
              shape: { dim: [{ dimValue: CHANNELS }] },
            },
          },
        }),
      )
    }
    nodes.push(node)
  }
  if (found.size !== LAYERS || [...found].some((x) => x < 0 || x >= LAYERS))
    throw new Error(
      'Expected 36 Qwen3 MLP layers; refusing to instrument an unknown graph',
    )
  graph.node = nodes.map((n) => onnx.NodeProto.create(n))
  return onnx.ModelProto.encode(model).finish()
}
