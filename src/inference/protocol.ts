export type ActivationFrame = {
  values: Float32Array
  step: number
  tokens: number
}
export type Message = { role: 'user' | 'assistant'; content: string }
export type Neuron = {
  layer: number
  channel: number
  score: number
  targetMean?: number
  controlMean?: number
  effect?: number
  consistency?: number
  activationRate?: number
  p?: number
  q?: number
}
export type Trial = {
  count: number
  targetDelta: number
  controlDelta: number
  samples: {
    prompt: string
    group: 'target' | 'control'
    continuation: string
    delta: number
  }[]
}
export type Report = {
  concept: string
  controls: string[]
  controlGeneration: string
  statistics: { tested: number; significant: number; pairs: number }
  candidates: Neuron[]
  probes: {
    prompt: string
    continuation: string
    group: string
    split: string
  }[]
}
export type Request =
  | { type: 'generate'; messages: Message[] }
  | { type: 'analyze'; concept: string }
  | { type: 'mask'; neurons: Neuron[] }
  | { type: 'evaluate'; threshold: number }
  | { type: 'stop' }
export type Reply =
  | { type: 'activations'; frame: ActivationFrame }
  | { type: 'status'; text: string }
  | { type: 'chunk'; text: string }
  | { type: 'done' }
  | { type: 'report'; report: Report }
  | { type: 'evaluation'; threshold: number; trial: Trial }
  | { type: 'error'; text: string }
