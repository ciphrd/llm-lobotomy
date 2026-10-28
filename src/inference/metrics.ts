import type { Neuron } from './protocol'

export function rankChannels(
  target: number[][],
  control: number[][],
): Neuron[] {
  const ranked: Neuron[] = []
  for (let layer = 0; layer < target.length; layer++) {
    const t = target[layer],
      c = control[layer]
    const scale = t.reduce((a, b) => a + b, 0) / t.length + 1e-8
    for (let channel = 0; channel < t.length; channel++) {
      const score =
        ((t[channel] - c[channel]) / (t[channel] + c[channel] + 1e-8)) *
        Math.log1p(t[channel] / scale)
      if (Number.isFinite(score) && score > 0)
        ranked.push({ layer, channel, score })
    }
  }
  return ranked.sort((a, b) => b.score - a.score)
}

/** Mean teacher-forced continuation NLL, measured in nats/token. */
export function continuationLoss(
  logits: Float32Array,
  vocabulary: number,
  tokens: number[],
  promptLength: number,
): number {
  if (tokens.length <= promptLength || promptLength < 1)
    throw new Error('Empty continuation')
  let loss = 0
  for (let pos = promptLength; pos < tokens.length; pos++) {
    const offset = (pos - 1) * vocabulary
    let max = -Infinity
    for (let j = 0; j < vocabulary; j++) max = Math.max(max, logits[offset + j])
    if (!Number.isFinite(max) || !Number.isFinite(logits[offset + tokens[pos]]))
      throw new Error(
        `Invalid logits at token ${pos}, id ${tokens[pos]}, vocab ${vocabulary}, buffer ${logits.length}, max ${max}`,
      )
    let sum = 0
    for (let j = 0; j < vocabulary; j++)
      sum += Math.exp(logits[offset + j] - max)
    loss += max + Math.log(sum) - logits[offset + tokens[pos]]
  }
  const result = loss / (tokens.length - promptLength)
  if (!Number.isFinite(result))
    throw new Error('Non-finite likelihood; the experiment is invalid')
  return result
}
