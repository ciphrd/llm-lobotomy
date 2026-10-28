import { useEffect, useRef, useState, type RefObject } from 'react'
import type { ActivationFrame } from './inference/protocol'
import { CHANNELS, LAYERS } from './inference/topology'
import { createRenderer } from './visualization/renderer'

export function ActivationView({
  frame,
  busy,
}: {
  frame: RefObject<ActivationFrame | null>
  busy: boolean
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const camera = useRef({
    yaw: -0.65,
    pitch: -0.3,
    distance: 12,
    layer: -1,
    gain: 10,
  })
  const [layer, setLayer] = useState(-1)
  const [channel, setChannel] = useState(0)
  const [gain, setGain] = useState(10)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  const [sample, setSample] = useState<ActivationFrame | null>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)
  useEffect(() => {
    let disposed = false
    let animation = 0
    let renderer: Awaited<ReturnType<typeof createRenderer>> | undefined
    let previous: ActivationFrame | null | undefined
    let failed = false
    const fail = (message: string) => {
      failed = true
      if (!disposed) setError(message)
    }
    const controller = new AbortController()
    void createRenderer(canvas.current!, fail, controller.signal)
      .then((result) => {
        if (disposed) {
          result.dispose()
          return
        }
        renderer = result
        setReady(true)
        const draw = () => {
          if (disposed || failed) return
          try {
            if (frame.current !== previous) {
              previous = frame.current
              renderer!.upload(
                previous?.values ?? new Float32Array(LAYERS * CHANNELS),
              )
            }
            const c = camera.current
            renderer!.draw(c.yaw, c.pitch, c.distance, c.layer, c.gain)
            animation = requestAnimationFrame(draw)
          } catch (e) {
            fail(e instanceof Error ? e.message : String(e))
          }
        }
        draw()
      })
      .catch((e: unknown) => fail(e instanceof Error ? e.message : String(e)))
    // Text updates are independent of the full-rate GPU uploads.
    const timer = window.setInterval(() => setSample(frame.current), 200)
    const element = canvas.current!
    const wheel = (event: WheelEvent) => {
      event.preventDefault()
      camera.current.distance = Math.min(
        30,
        Math.max(6, camera.current.distance * Math.exp(event.deltaY * 0.001)),
      )
    }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => {
      disposed = true
      controller.abort()
      cancelAnimationFrame(animation)
      clearInterval(timer)
      element.removeEventListener('wheel', wheel)
      renderer?.dispose()
    }
  }, [frame])
  return (
    <section
      className="activation-view"
      aria-label="Live 3D neuron activations"
    >
      <div className="activation-heading">
        <strong>Inside the layers</strong>
        <span>
          {sample ? (busy ? 'Live' : 'Last measurement') : 'Awaiting inference'}{' '}
          · WebGPU
        </span>
      </div>
      <div className="activation-stage">
        <canvas
          ref={canvas}
          aria-label="3D MLP layers. Drag to rotate; scroll or use the zoom buttons to zoom."
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId)
            drag.current = { x: e.clientX, y: e.clientY }
          }}
          onPointerMove={(e) => {
            if (!drag.current) return
            camera.current.yaw += (e.clientX - drag.current.x) * 0.006
            camera.current.pitch = Math.max(
              -1.4,
              Math.min(
                1.4,
                camera.current.pitch + (e.clientY - drag.current.y) * 0.006,
              ),
            )
            drag.current = { x: e.clientX, y: e.clientY }
          }}
          onPointerUp={() => {
            drag.current = null
          }}
          onPointerCancel={() => {
            drag.current = null
          }}
          onLostPointerCapture={() => {
            drag.current = null
          }}
        />
        {(!ready || error) && (
          <p className="activation-overlay" role="status">
            {error || 'Initializing WebGPU…'}
          </p>
        )}
        <div className="activation-caption">
          36 layers · 9,728 channels each · 350,208 neurons
        </div>
      </div>
      <div className="activation-controls">
        <label>
          Layer{' '}
          <select
            value={layer}
            onChange={(e) => {
              const value = Number(e.target.value)
              setLayer(value)
              camera.current.layer = value
            }}
          >
            <option value={-1}>All layers</option>
            {Array.from({ length: LAYERS }, (_, i) => (
              <option key={i} value={i}>
                Layer {i}
              </option>
            ))}
          </select>
        </label>
        <label>
          Brightness{' '}
          <input
            type="range"
            min="1"
            max="100"
            value={gain}
            onChange={(e) => {
              setGain(Number(e.target.value))
              camera.current.gain = Number(e.target.value)
            }}
          />
        </label>
        <button
          type="button"
          aria-label="Zoom in"
          onClick={() => {
            camera.current.distance = Math.max(6, camera.current.distance - 1)
          }}
        >
          +
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          onClick={() => {
            camera.current.distance = Math.min(30, camera.current.distance + 1)
          }}
        >
          −
        </button>
        <button
          type="button"
          onClick={() => {
            camera.current.yaw = -0.65
            camera.current.pitch = -0.3
            camera.current.distance = 12
          }}
        >
          Reset view
        </button>
      </div>
      <p className="activation-key">
        <span />
        Low → high magnitude · Drag to orbit · Scroll to zoom
      </p>
      <p className="status">
        Each point is one MLP channel, ordered by channel index within each
        layer. Values are absolute activations after the current mask; prompt
        and scoring passes average across tokens.{' '}
        {sample
          ? `Step ${sample.step} · ${sample.tokens} token${sample.tokens === 1 ? '' : 's'} processed.`
          : 'Send a message or analyze a concept to see real measurements.'}
      </p>
      {layer >= 0 && (
        <label className="activation-inspector">
          Channel{' '}
          <input
            type="number"
            min={0}
            max={CHANNELS - 1}
            value={channel}
            onChange={(e) =>
              setChannel(
                Math.max(
                  0,
                  Math.min(
                    CHANNELS - 1,
                    Math.trunc(Number(e.target.value)) || 0,
                  ),
                ),
              )
            }
          />{' '}
          <output>
            Layer {layer}, channel {channel}:{' '}
            {sample
              ? sample.values[layer * CHANNELS + channel].toPrecision(5)
              : 'no measurement'}
          </output>
        </label>
      )}
    </section>
  )
}
