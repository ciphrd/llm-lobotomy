import { test } from 'node:test'
import assert from 'node:assert/strict'
import { AnalysisCache } from '../src/inference/analysis-cache.ts'
import type { Report } from '../src/inference/protocol.ts'

function memoryStorage() {
  const stores = new Map<string, Map<string, Response>>()
  const storage = {
    keys: async () => [...stores.keys()],
    delete: async (name: string) => stores.delete(name),
    open: async (name: string) => {
      if (!stores.has(name)) stores.set(name, new Map())
      const entries = stores.get(name)!
      return {
        keys: async () => [...entries.keys()],
        match: async (key: string) => entries.get(key)?.clone(),
        put: async (key: string, value: Response) => {
          entries.set(key, value.clone())
        },
        delete: async (key: string) => entries.delete(key),
      }
    },
  } as unknown as CacheStorage
  return { storage, stores }
}
const report: Report = {
  concept: 'cat',
  controls: ['dog', 'lion', 'tiger'],
  controlSource: 'generated',
  controlGeneration: '[]',
  statistics: { tested: 350208, significant: 1, pairs: 64 },
  candidates: [{ layer: 0, channel: 1, score: 2 }],
  probes: [],
}

test('analysis persists across cache instances and trims terms without merging case', async () => {
  const { storage } = memoryStorage()
  const first = new AnalysisCache('1:model-a', storage)
  assert.equal(await first.get('cat'), null)
  assert.ok(await first.put(report))
  const next = new AnalysisCache('1:model-a', storage)
  assert.deepEqual((await next.get(' cat '))?.report, report)
  assert.equal(await next.get('Cat'), null)
  assert.equal(await next.get('dog'), null)
})

test('version or model changes delete old analysis caches but preserve model downloads', async () => {
  const { storage, stores } = memoryStorage()
  await storage.open('qwen-onnx-model-downloads')
  await new AnalysisCache('1:model-a', storage).put(report)
  const upgraded = new AnalysisCache('2:model-a', storage)
  assert.equal(await upgraded.get('cat'), null)
  assert.ok(stores.has('qwen-onnx-model-downloads'))
  assert.ok(!stores.has('qwen-analysis-1%3Amodel-a'))
  await upgraded.put(report)
  assert.equal(await new AnalysisCache('2:model-b', storage).get('cat'), null)
  assert.ok(!stores.has('qwen-analysis-2%3Amodel-a'))
})

test('storage failures do not prevent analysis; invalid entries are cache misses', async () => {
  const denied = new AnalysisCache('1', {
    keys: async () => {
      throw new Error('denied')
    },
  } as unknown as CacheStorage)
  assert.equal(await denied.get('cat'), null)
  assert.equal(await denied.put(report), null)
  const { storage, stores } = memoryStorage()
  const cache = new AnalysisCache('1', storage)
  await cache.put(report)
  stores
    .get('qwen-analysis-1')!
    .set(
      `https://analysis-cache.invalid/term/${encodeURIComponent(JSON.stringify(['cat', null]))}`,
      new Response('{broken'),
    )
  assert.equal(await cache.get('cat'), null)
  const store = await storage.open('qwen-analysis-1')
  store.put = async () => {
    throw new Error('quota')
  }
  const quotaStorage = {
    keys: async () => ['qwen-analysis-1'],
    open: async () => store,
  } as unknown as CacheStorage
  assert.equal(await new AnalysisCache('1', quotaStorage).put(report), null)
})

test('provided control sets have separate cache entries from automatic and other sets', async () => {
  const { storage } = memoryStorage()
  const cache = new AnalysisCache('3', storage)
  await cache.put(report)
  const provided: Report = { ...report, controlSource: 'provided' }
  assert.equal(await cache.get('cat', provided.controls), null)
  await cache.put(provided)
  assert.deepEqual(
    (await cache.get('cat', provided.controls))?.report,
    provided,
  )
  assert.equal((await cache.get('cat'))?.report.controlSource, 'generated')
  assert.equal(await cache.get('cat', ['horse', 'lion', 'tiger']), null)
})

test('listing distinguishes control sets, ignores corrupt entries, and supports old v3 reports', async () => {
  const { storage, stores } = memoryStorage()
  const cache = new AnalysisCache('3', storage)
  assert.deepEqual(await cache.list(), [])
  await cache.put(report)
  await cache.put({ ...report, controlSource: 'provided' })
  let entries = await cache.list()
  assert.equal(entries.length, 2)
  assert.deepEqual(
    new Set(entries.map((e) => e.controlSource)),
    new Set(['generated', 'provided']),
  )
  assert.ok(entries.every((e) => !('candidates' in e)))
  const records = stores.get('qwen-analysis-3')!
  records.set(
    'legacy',
    new Response(
      JSON.stringify({
        version: '3',
        savedAt: '2020-01-01',
        report: { ...report, concept: 'legacy' },
      }),
    ),
  )
  records.set('broken', new Response('{'))
  entries = await cache.list()
  assert.equal(entries.length, 3)
  assert.equal(entries.at(-1)?.concept, 'legacy')
})
