import { writeMask } from '../src/inference/mask.ts'
import { test } from 'node:test'
import assert from 'node:assert/strict'
import proto from 'onnx-proto'
const { onnx } = proto
import { instrumentGraph, LAYERS, CHANNELS } from '../src/inference/graph.ts'
import { continuationLoss, rankChannels } from '../src/inference/metrics.ts'

test('inserts a dynamic mask on every down projection without touching weights', () => {
  const graph = onnx.ModelProto.create({
    graph: {
      node: Array.from({ length: LAYERS }, (_, layer) => ({
        name: `/model/layers.${layer}/mlp/down_proj/MatMul_Q4`,
        opType: 'MatMulNBits',
        input: [`activation_${layer}`, `weight_${layer}`],
        output: [`out_${layer}`],
        attribute: [{ name: 'K', type: 2, i: CHANNELS }],
      })),
      initializer: [
        { name: 'weight_0', dataType: 1, dims: [1], floatData: [42] },
      ],
    },
  })
  const result = onnx.ModelProto.decode(
    instrumentGraph(onnx.ModelProto.encode(graph).finish()),
  ).graph!
  assert.equal(result.input?.length, LAYERS)
  assert.equal(result.output?.length, LAYERS)
  assert.equal(result.node?.filter((n) => n.opType === 'Mul').length, LAYERS)
  for (let layer = 0; layer < LAYERS; layer++) {
    const projection = result.node!.find(
      (n) => n.name === `/model/layers.${layer}/mlp/down_proj/MatMul_Q4`,
    )!
    assert.deepEqual(projection.input, [
      `masked_mlp_${layer}`,
      `weight_${layer}`,
    ])
    assert.deepEqual(
      result.node!.find((n) => n.name === `masked_mlp_${layer}`)!.input,
      [`activation_${layer}`, `ablation_mask_${layer}`],
    )
    assert.equal(result.input![layer].type!.tensorType!.elemType, 10)
  }
  assert.equal(result.initializer![0].floatData![0], 42)
})

test('refuses unknown graph structures', () => {
  assert.throws(
    () =>
      instrumentGraph(onnx.ModelProto.encode({ graph: { node: [] } }).finish()),
    /Expected 36/,
  )
})

test('continuation likelihood uses the preceding token logits, excluding prompt loss', () => {
  const logits = new Float32Array([1000, 1001, 0, 0, 0, 1000])
  const loss = continuationLoss(logits, 2, [0, 1, 0], 1)
  assert.ok(
    Math.abs(loss - (Math.log1p(Math.exp(-1)) + Math.log(2)) / 2) < 1e-8,
  )
  assert.throws(() => continuationLoss(logits, 2, [0], 1), /Empty continuation/)
})

test('ranking excludes channels more active for controls and handles all-zero activations', () => {
  const ranked = rankChannels([[10, 1, 0]], [[1, 10, 0]])
  assert.equal(ranked.length, 1)
  assert.equal(ranked[0].channel, 0)
  assert.deepEqual(rankChannels([[0]], [[0]]), [])
})

test('mask updates write half bits through the underlying buffer, not numeric float16 values', () => {
  const bytes = new Uint8Array(12)
  const view = bytes.subarray(2, 10)
  writeMask(view, 4, [1])
  assert.deepEqual(
    [...new Uint16Array(bytes.buffer)],
    [0, 0x3c00, 0, 0x3c00, 0x3c00, 0],
  )
  writeMask(view, 4, [])
  assert.deepEqual(
    [...new Uint16Array(bytes.buffer)].slice(1, 5),
    [0x3c00, 0x3c00, 0x3c00, 0x3c00],
  )
  assert.throws(() => writeMask(view, 4, [4]), /Invalid channel/)
})
