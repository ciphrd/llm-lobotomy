import { BrowserModel } from './inference/runtime'
import { aboveThreshold } from './inference/statistics'
import type { Report, Request, Reply } from './inference/protocol'
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
    if (!model) {
      model = new BrowserModel((frame) => {
        self.postMessage({ type: 'activations', frame } satisfies Reply, {
          transfer: [frame.values.buffer],
        })
      })
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
      const report = await model.analyze(concept, status)
      lastReport = report
      reply({ type: 'report', report })
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
