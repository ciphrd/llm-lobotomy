import type { Neuron } from './protocol'

/** One-sided exact sign test, dropping ties. */
export function signP(positive: number, count: number): number {
  if (!count || positive === 0) return 1
  let coefficient = 1,
    tail = 0
  for (let k = 0; k <= count; k++) {
    if (k >= positive) tail += coefficient
    coefficient *= (count - k) / (k + 1)
  }
  return Math.min(1, tail / 2 ** count)
}

export function adjustBH(p: Float64Array): Float64Array {
  const order = Array.from(p.keys()).sort((a, b) => p[a] - p[b])
  const q = new Float64Array(p.length)
  let previous = 1
  for (let i = order.length - 1; i >= 0; i--) {
    previous = Math.min(previous, (p[order[i]] * p.length) / (i + 1))
    q[order[i]] = previous
  }
  return q
}

/** Each template is one paired observation; controls are averaged within that template. */
export function analyzeProfiles(
  target: Float32Array[],
  control: Float32Array[],
  width: number,
) {
  const n = target.length,
    total = target[0]?.length ?? 0
  if (
    n < 2 ||
    control.length !== n ||
    !total ||
    total % width ||
    [...target, ...control].some((a) => a.length !== total)
  )
    throw new Error('Invalid paired activation profiles')
  const p = new Float64Array(total),
    records: Neuron[] = []
  for (let j = 0; j < total; j++) {
    let tSum = 0,
      cSum = 0,
      cSq = 0,
      sum = 0,
      sq = 0,
      wins = 0,
      nonTies = 0
    for (let i = 0; i < n; i++) {
      const t = target[i][j],
        c = control[i][j],
        d = t - c
      if (!Number.isFinite(t) || !Number.isFinite(c))
        throw new Error('Non-finite activation profile')
      tSum += t
      cSum += c
      cSq += c * c
      sum += d
      sq += d * d
      if (Math.abs(d) > 1e-8) {
        nonTies++
        if (d > 0) wins++
      }
    }
    const mean = sum / n,
      variance = Math.max(0, (sq - (sum * sum) / n) / (n - 1))
    const controlMean = cSum / n,
      controlSD = Math.sqrt(Math.max(0, (cSq - (cSum * cSum) / n) / (n - 1)))
    const effect = mean / Math.sqrt(variance + 1e-12)
    const consistency = wins / n
    const activationRate =
      target.filter((a) => a[j] > controlMean + controlSD).length / n
    p[j] = mean > 0 ? signP(wins, nonTies) : 1
    records.push({
      layer: Math.floor(j / width),
      channel: j % width,
      score: effect * consistency,
      targetMean: tSum / n,
      controlMean,
      effect,
      consistency,
      activationRate,
      p: p[j],
      q: 1,
    })
  }
  const q = adjustBH(p)
  for (let j = 0; j < total; j++) records[j].q = q[j]
  const significant = records.filter(isQualified).length
  const candidates = records
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
  return { candidates, tested: total, significant, pairs: n }
}

export function isQualified(neuron: Neuron): boolean {
  return (
    (neuron.q ?? 1) <= 0.05 &&
    (neuron.consistency ?? 0) >= 0.75 &&
    (neuron.effect ?? 0) >= 0.5
  )
}

/** Strict threshold, with no top-k limit or hidden significance filter. */
export function aboveThreshold(neurons: Neuron[], threshold: number): Neuron[] {
  if (!Number.isFinite(threshold) || threshold < 0)
    throw new Error('Threshold must be a finite nonnegative number')
  return neurons.filter((n) => Number.isFinite(n.score) && n.score > threshold)
}
