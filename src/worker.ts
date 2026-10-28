import {
  AnalysisCache,
  ANALYSIS_METHOD_VERSION,
} from './inference/analysis-cache'
import { MODEL_ID, REVISION } from './inference/graph'
import { BrowserModel } from './inference/runtime'
import { parseControls } from './inference/probes'
import { aboveThreshold } from './inference/statistics'
import type { Report, Request, Reply } from './inference/protocol'
const analysisCache = new AnalysisCache(
  `${ANALYSIS_METHOD_VERSION}:${MODEL_ID}:${REVISION}:q4f16`,
  self.caches,
)
let model: BrowserModel | null = null
let busy = false
let lastReport: Report | null = null
const reply = (message: Reply) => self.postMessage(message)
const status = (text: string) => reply({ type: 'status', text })
self.onmessage = async ({ data }: MessageEvent<Request>) => {
  if (data.type === 'stop') {
    model?.stop.interrupt()
    return
  }
  if (busy) return
  busy = true
  try {
    await analysisCache.initialize()
    if (data.type === 'list-analyses') {
      reply({ type: 'analyses', entries: await analysisCache.list() })
      reply({ type: 'done' })
      return
    }
    if (data.type === 'analyze') {
      const concept = data.concept.trim()
      if (!concept || concept.length > 100)
        throw new Error('Enter a concept between 1 and 100 characters.')
      if (data.controls !== undefined)
        data.controls = parseControls(JSON.stringify(data.controls), concept)
      lastReport = null
      model?.setMask([])
      status('Checking analysis cache…')
      const cached = await analysisCache.get(concept, data.controls)
      if (cached) {
        lastReport = cached.report
        reply({
          type: 'report',
          report: cached.report,
          cache: {
            hit: true,
            version: ANALYSIS_METHOD_VERSION,
            savedAt: cached.savedAt,
          },
        })
        reply({ type: 'analyses', entries: await analysisCache.list() })
        reply({ type: 'done' })
        return
      }
      if (data.cachedOnly) {
        reply({ type: 'analyses', entries: await analysisCache.list() })
        throw new Error(
          'This cached analysis is no longer available. Select Analyze to run it again.',
        )
      }
    }
    if (!model) {
      model = new BrowserModel(
        (frame) => {
          self.postMessage({ type: 'activations', frame } satisfies Reply, {
            transfer: [frame.values.buffer],
          })
        },
        (disabled) => {
          self.postMessage({ type: 'mask-state', disabled } satisfies Reply, {
            transfer: [disabled.buffer],
          })
        },
      )
      status('Loading Qwen3 4B with Transformers.js…')
      try {
        await model.load(status)
      } catch (error) {
        await model.dispose()
        model = null
        throw error
      }
    }
    if (data.type === 'generate') {
      status('Generating…')
      await model.generate(data.messages, (text) =>
        reply({ type: 'chunk', text }),
      )
    } else if (data.type === 'analyze') {
      const concept = data.concept.trim()
      if (!concept || concept.length > 100)
        throw new Error('Enter a concept between 1 and 100 characters.')
      lastReport = null
      const report = await model.analyze(concept, status, data.controls)
      lastReport = report
      const savedAt = await analysisCache.put(report)
      reply({ type: 'analyses', entries: await analysisCache.list() })
      reply({
        type: 'report',
        report,
        cache: { hit: false, version: ANALYSIS_METHOD_VERSION, savedAt },
      })
    } else if (data.type === 'evaluate') {
      if (!lastReport) throw new Error('Analyze a concept first.')
      const trial = await model.evaluate(
        lastReport,
        aboveThreshold(lastReport.candidates, data.threshold),
        status,
      )
      reply({ type: 'evaluation', threshold: data.threshold, trial })
    } else if (data.type === 'mask') model.setMask(data.neurons)
    reply({ type: 'done' })
  } catch (error) {
    reply({
      type: 'error',
      text: error instanceof Error ? error.message : String(error),
    })
  } finally {
    busy = false
  }
}
