import { useEffect, useMemo, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message, Reply, Request, Report } from './inference/protocol'
import { PROFILE_TEMPLATES, TEST_TEMPLATES } from './inference/probes'
import { aboveThreshold, isQualified } from './inference/statistics'
import type { Trial } from './inference/protocol'
import './App.css'
import { ActivationView } from './ActivationView'
import type { ActivationFrame } from './inference/protocol'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [concept, setConcept] = useState('')
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [report, setReport] = useState<Report | null>(null)
  const [applied, setApplied] = useState<{
    threshold: number
    count: number
  } | null>(null)
  const [threshold, setThreshold] = useState(1)
  const [evaluation, setEvaluation] = useState<{
    threshold: number
    trial: Trial
  } | null>(null)
  const masked = applied !== null && applied.count > 0
  const selected = useMemo(
    () => (report ? aboveThreshold(report.candidates, threshold) : []),
    [report, threshold],
  )
  const qualifiedCount = useMemo(
    () => selected.filter(isQualified).length,
    [selected],
  )
  const maximum = report?.candidates[0]?.score ?? 1
  const activationFrame = useRef<ActivationFrame | null>(null)
  const worker = useRef<Worker | null>(null)
  const locked = useRef(false)
  const pendingMask = useRef<
    { threshold: number; count: number } | null | undefined
  >(undefined)
  const ready = useRef(false)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    locked.current = false
    ready.current = false
    setBusy(false)
    setStatus('')
    setApplied(null)
    return () => {
      worker.current?.terminate()
      worker.current = null
    }
  }, [])
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function dispatch(request: Request) {
    if (locked.current) return
    locked.current = true
    setBusy(true)
    setError('')
    pendingMask.current =
      request.type === 'mask'
        ? request.neurons.length
          ? { threshold, count: request.neurons.length }
          : null
        : undefined
    activationFrame.current = null
    if (!worker.current) {
      worker.current = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      })
      worker.current.onmessage = ({ data }: MessageEvent<Reply>) => {
        if (data.type === 'activations') {
          activationFrame.current = data.frame
          return
        }
        if (data.type === 'status') {
          setStatus(data.text)
          if (/^(Loading|Preparing|Instrumenting)/.test(data.text))
            ready.current = false
          if (
            /^(Generating|Profiling|Computing|Baseline|Testing)/.test(data.text)
          )
            ready.current = true
        }
        if (data.type === 'chunk')
          setMessages((previous) => {
            const last = previous.at(-1)
            return last?.role === 'assistant'
              ? [
                  ...previous.slice(0, -1),
                  { ...last, content: last.content + data.text },
                ]
              : previous
          })
        if (data.type === 'report') {
          setReport(data.report)
          setThreshold(1)
          setEvaluation(null)
        }
        if (data.type === 'evaluation')
          setEvaluation({ threshold: data.threshold, trial: data.trial })
        if (data.type === 'done' || data.type === 'error') {
          locked.current = false
          setBusy(false)
          setStatus('')
          if (data.type === 'error') setError(data.text)
          if (data.type === 'done') ready.current = true
          if (data.type === 'done' && pendingMask.current !== undefined) {
            setApplied(pendingMask.current)
            setMessages([])
          }
          pendingMask.current = undefined
          setMessages((previous) =>
            previous.at(-1)?.role === 'assistant' && !previous.at(-1)?.content
              ? previous.slice(0, -1)
              : previous,
          )
        }
      }
      worker.current.onerror = (event) => {
        setError(event.message || 'The model worker failed. Try again.')
        worker.current?.terminate()
        worker.current = null
        ready.current = false
        locked.current = false
        setBusy(false)
        setStatus('')
        setApplied(null)
      }
    }
    worker.current.postMessage(request)
  }

  function send() {
    if (!input.trim() || locked.current) return
    const history: Message[] = [
      ...messages,
      { role: 'user', content: input.trim() },
    ]
    setMessages([...history, { role: 'assistant', content: '' }])
    setInput('')
    dispatch({ type: 'generate', messages: history })
  }

  function stop() {
    if (!ready.current) {
      worker.current?.terminate()
      worker.current = null
      locked.current = false
      setBusy(false)
      setStatus('')
      setApplied(null)
      setMessages((previous) => previous.filter((m) => m.content))
    } else worker.current?.postMessage({ type: 'stop' } satisfies Request)
  }

  function apply(enabled: boolean) {
    if (locked.current || !report) return
    dispatch({
      type: 'mask',
      neurons: enabled
        ? selected.map(({ layer, channel, score }) => ({
            layer,
            channel,
            score,
          }))
        : [],
    })
  }

  function exportReport() {
    if (!report) return
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            {
              model: 'onnx-community/Qwen3-4B-ONNX',
              revision: 'ca652ef636e8b121ff4a1607c37ab578cb866987',
              method: `Generated related controls; ${report.statistics.pairs} paired prompt-only activation profiles; relatedness = paired effect size times consistency; all positive scores retained without a channel cap; optional separate continuation-loss test. Exploratory statistics, not proof of concept erasure.`,
              ...report,
              threshold,
              selected,
              applied,
              evaluation,
            },
            null,
            2,
          ),
        ],
        { type: 'application/json' },
      ),
    )
    const a = document.createElement('a')
    a.href = url
    a.download = 'ablation-report.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const delta = (n: number) => `${n >= 0 ? '+' : ''}${n.toFixed(3)}`

  return (
    <main className="workspace">
      <section className="chat" aria-label="Inference">
        <form
          className="concept"
          onSubmit={(e) => {
            e.preventDefault()
            if (!concept.trim() || locked.current) return
            setReport(null)
            setEvaluation(null)
            setApplied(null)
            setMessages([])
            dispatch({ type: 'analyze', concept: concept.trim() })
          }}
        >
          <input
            aria-label="Concept to ablate"
            placeholder="Concept to ablate…"
            value={concept}
            maxLength={100}
            onChange={(e) => setConcept(e.target.value)}
            disabled={busy}
          />
          <button
            disabled={busy || !concept.trim()}
            title={`${PROFILE_TEMPLATES.length} profiling templates, each with target and three controls`}
          >
            Analyze
          </button>
        </form>
        {report && (
          <section className="report" aria-label="Ablation results">
            <p>
              {masked
                ? `${applied.count.toLocaleString()} neurons masked (score > ${applied.threshold}).`
                : 'Original model active.'}
            </p>
            <label className="threshold-label" htmlFor="threshold">
              Relatedness threshold
            </label>
            <div className="threshold-controls">
              <input
                id="threshold"
                type="range"
                min="0"
                max={Math.log1p(maximum)}
                step="any"
                value={Math.log1p(Math.min(threshold, maximum))}
                disabled={busy || !report.candidates.length}
                onChange={(e) =>
                  setThreshold(
                    Number(e.target.value) === Math.log1p(maximum)
                      ? maximum
                      : Math.expm1(Number(e.target.value)),
                  )
                }
                aria-valuetext={`${threshold.toPrecision(4)}; ${selected.length} neurons selected`}
              />
              <input
                aria-label="Exact relatedness threshold"
                type="number"
                min="0"
                step="any"
                value={threshold}
                disabled={busy}
                onChange={(e) => {
                  const value = e.target.valueAsNumber
                  if (Number.isFinite(value) && value >= 0) setThreshold(value)
                }}
              />
            </div>
            <p aria-live="polite">
              {selected.length.toLocaleString()} neurons with score &gt;{' '}
              {threshold.toPrecision(4)} selected. No channel limit.
            </p>
            <p className="status">
              Score = paired effect × consistency, not a probability. Lower
              thresholds select more neurons. {qualifiedCount.toLocaleString()}{' '}
              selected neurons pass the statistical checks; those checks do not
              restrict the mask. Slider changes take effect when you apply them.
            </p>
            <p>Related controls: {report.controls.join(', ')}.</p>
            <p className="status">
              {report.statistics.pairs} matched templates ·{' '}
              {report.statistics.tested.toLocaleString()} channels measured ·{' '}
              {report.statistics.significant.toLocaleString()} qualified.
              Prompt-only mean absolute activations; each target is compared
              with the mean of its three controls.
            </p>
            {evaluation && (
              <p className="status">
                Tested score &gt; {evaluation.threshold.toPrecision(4)} (
                {evaluation.trial.count.toLocaleString()} neurons): target loss
                Δ {delta(evaluation.trial.targetDelta)}, control Δ{' '}
                {delta(evaluation.trial.controlDelta)} nats/token. Higher loss
                means the original answer became less likely, not verified
                forgetting. Tested{' '}
                {
                  evaluation.trial.samples.filter((s) => s.group === 'target')
                    .length
                }{' '}
                target prompts and{' '}
                {
                  evaluation.trial.samples.filter((s) => s.group === 'control')
                    .length
                }{' '}
                control prompts.
                {evaluation.threshold !== threshold &&
                  ' This test belongs to a different threshold.'}
              </p>
            )}
            <div className="actions">
              <button type="button" className="clear" onClick={exportReport}>
                Export results
              </button>
              <button
                disabled={busy || !selected.length}
                onClick={() => dispatch({ type: 'evaluate', threshold })}
              >
                Test threshold
              </button>
              {masked && (
                <button disabled={busy} onClick={() => apply(false)}>
                  Restore original
                </button>
              )}
              <button
                disabled={
                  busy ||
                  (applied?.threshold === threshold &&
                    applied.count === selected.length)
                }
                onClick={() => apply(true)}
              >
                Apply {selected.length.toLocaleString()} neurons
              </button>
            </div>
            <details>
              <summary>Prompts and neurons</summary>
              <p className="status">
                Profiling uses {report.statistics.pairs} template pairs. Every
                positively scored channel is available; no top-k cap or hidden
                significance filter is applied. Statistical checks: exploratory
                BH q ≤ 0.05, positive difference in ≥75% of templates, paired
                standardized effect ≥0.5. Related templates are not independent
                random samples, so q-values are screening evidence, not proof.
                Prompts name the concept; lexical effects remain a confound.
                Generated controls may be semantically imperfect. Test threshold
                compares original versus masked continuation loss on{' '}
                {TEST_TEMPLATES.length} separate templates for the target and
                three controls, then restores your previously applied mask.
                Repeated threshold tests are exploratory, not an independent
                confirmation of forgetting.
              </p>
              {evaluation && (
                <details>
                  <summary>
                    Per-question loss changes (threshold{' '}
                    {evaluation.threshold.toPrecision(4)})
                  </summary>
                  {evaluation.trial.samples.map((s, i) => (
                    <p key={i}>
                      <small>
                        {s.group} · Δ {delta(s.delta)} nats/token
                      </small>
                      <br />
                      {s.prompt} <em>{s.continuation}</em>
                    </p>
                  ))}
                </details>
              )}
              <p>
                Control-generation response:{' '}
                <code>{report.controlGeneration}</code>
              </p>
              {report.probes.map((p, i) => (
                <p key={i}>
                  <small>
                    {p.group} / {p.split}
                  </small>
                  <br />
                  {p.prompt} <em>{p.continuation}</em>
                </p>
              ))}
              <pre>
                {report.candidates
                  .slice(0, 32)
                  .map(
                    (n) =>
                      `L${n.layer} C${n.channel}: score ${n.score.toFixed(3)}, effect ${n.effect?.toFixed(2)}, consistency ${Math.round((n.consistency ?? 0) * 100)}%, strong activation ${Math.round((n.activationRate ?? 0) * 100)}%, q=${n.q?.toExponential(2)}`,
                  )
                  .join('\n')}
              </pre>
            </details>
          </section>
        )}
        <section className="messages" aria-label="Conversation">
          {messages.map((message, i) => (
            <article
              key={i}
              className={`message ${message.role}`}
              aria-label={message.role === 'user' ? 'You' : 'Qwen'}
            >
              {message.content ? (
                <Markdown remarkPlugins={[remarkGfm]}>
                  {message.role === 'assistant'
                    ? message.content.replace(/<think>\s*<\/think>\s*/g, '')
                    : message.content}
                </Markdown>
              ) : (
                <span className="status">…</span>
              )}
            </article>
          ))}
          <div ref={bottom} />
        </section>
        <div className="input-area">
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {status && (
            <p className="status" role="status">
              {status}
            </p>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault()
              send()
            }}
          >
            <textarea
              aria-label="Message"
              placeholder={masked ? 'Message the ablated model…' : 'Message…'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={2}
              onKeyDown={(e) => {
                if (
                  e.key === 'Enter' &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <div className="actions">
              {messages.length > 0 && (
                <button
                  type="button"
                  className="clear"
                  onClick={() => setMessages([])}
                  disabled={busy}
                >
                  Clear
                </button>
              )}
              {busy ? (
                <button type="button" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button disabled={!input.trim()}>Send</button>
              )}
            </div>
          </form>
        </div>
      </section>
      <aside className="visualization-column" aria-label="Visualization">
        <ActivationView frame={activationFrame} busy={busy} />
      </aside>
    </main>
  )
}
