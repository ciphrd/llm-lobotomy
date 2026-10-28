import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  analyzeProfiles,
  signP,
  adjustBH,
  aboveThreshold,
} from '../src/inference/statistics.ts'
import {
  parseControls,
  PROFILE_TEMPLATES,
  SELECTION_TEMPLATES,
  TEST_TEMPLATES,
  renderProbe,
} from '../src/inference/probes.ts'

test('control parsing accepts JSON but rejects target variants, duplicates, and malformed lists', () => {
  assert.deepEqual(
    parseControls('```json\n["dog", "rabbit", "horse"]\n```', 'cat'),
    ['dog', 'rabbit', 'horse'],
  )
  for (const response of [
    '["cats","dog","horse"]',
    '["dog","dogs","horse"]',
    '["dog","horse"]',
    '[1,2,3]',
    'not JSON',
  ])
    assert.throws(() => parseControls(response, 'cat'))
})

test('sign test handles perfect consistency, chance, and ties', () => {
  assert.equal(signP(24, 24), 2 ** -24)
  assert.ok(Math.abs(signP(64, 64) / 2 ** -64 - 1) < 1e-12)
  assert.ok(signP(33, 64) > 0.4 && signP(33, 64) < 0.5)
  assert.equal(signP(0, 0), 1)
  assert.equal(signP(1, 2), 0.75)
})

test('BH corrects over the complete family of tests and preserves monotonicity', () => {
  const q = adjustBH(new Float64Array([0.01, 0.04, 0.03, 1]))
  assert.ok(Math.abs(q[0] - 0.04) < 1e-10)
  assert.ok(Math.abs(q[1] - (0.04 * 4) / 3) < 1e-10)
  assert.equal(q[1], q[2])
  assert.equal(q[3], 1)
})

test('paired profiles select recurring target effects, excluding general activation and reverse effects', () => {
  const targets = Array.from(
    { length: 24 },
    (_, i) => new Float32Array([10 + (i % 3), 100 + i, 1, 0]),
  )
  const controls = Array.from(
    { length: 24 },
    (_, i) => new Float32Array([1, 100 + i, 10, 0]),
  )
  const result = analyzeProfiles(targets, controls, 2)
  assert.equal(result.tested, 4)
  assert.equal(result.pairs, 24)
  assert.equal(result.candidates.length, 1)
  assert.equal(result.candidates[0].channel, 0)
  assert.equal(result.candidates[0].consistency, 1)
  assert.equal(result.candidates[0].activationRate, 1)
  assert.ok(result.candidates[0].q! < 0.05)
})

test('profiling, selection, and held-out templates are disjoint', () => {
  const all = [...PROFILE_TEMPLATES, ...SELECTION_TEMPLATES, ...TEST_TEMPLATES]
  assert.equal(new Set(all).size, all.length)
  assert.equal(PROFILE_TEMPLATES.length, 64)
  assert.equal(TEST_TEMPLATES.length, 12)
  for (const t of all) assert.ok(renderProbe(t, 'cat').includes('cat'))
})

test('profiles with no signal return no candidates; invalid data fails closed', () => {
  const zeros = Array.from({ length: 24 }, () => new Float32Array(4))
  assert.equal(analyzeProfiles(zeros, zeros, 2).candidates.length, 0)
  assert.throws(
    () =>
      analyzeProfiles(
        [new Float32Array([NaN]), new Float32Array([1])],
        [new Float32Array([1]), new Float32Array([1])],
        1,
      ),
    /Non-finite/,
  )
})

test('threshold selects every score strictly above it, without a 512 cap', () => {
  const neurons = Array.from({ length: 2000 }, (_, channel) => ({
    layer: 0,
    channel,
    score: channel / 1000,
  }))
  assert.equal(aboveThreshold(neurons, 0.5).length, 1499)
  assert.equal(aboveThreshold(neurons, 0).length, 1999)
  assert.equal(aboveThreshold(neurons, 1.999).length, 0)
  assert.deepEqual(aboveThreshold([], 0), [])
  for (const threshold of [-1, NaN, Infinity])
    assert.throws(() => aboveThreshold(neurons, threshold))
})

test('positive associations remain available even when statistical checks fail', () => {
  const targets = Array.from(
    { length: 24 },
    (_, i) => new Float32Array([i < 13 ? 2 : 0]),
  )
  const controls = targets.map(() => new Float32Array([1]))
  const result = analyzeProfiles(targets, controls, 1)
  assert.equal(result.significant, 0)
  assert.equal(result.candidates.length, 1)
  assert.equal(aboveThreshold(result.candidates, 0).length, 1)
})
