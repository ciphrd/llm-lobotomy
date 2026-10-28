import { env } from '@huggingface/transformers'
import { instrumentGraph, REVISION } from './graph'

/** Cache pristine downloads; instrument only the graph returned to the runtime. */
export async function configureCache(status: (text: string) => void) {
  env.allowLocalModels = false
  env.useBrowserCache = false
  env.useCustomCache = true
  let cache: Cache | null = null
  try {
    cache = await caches.open(`qwen-onnx-${REVISION}`)
  } catch {
    /* Inference still works without persistent caching. */
  }
  env.customCache = {
    async match(request: string) {
      const isGraph =
        request.includes(`/${REVISION}/onnx/model_q4f16.onnx`) &&
        !request.includes('.onnx_data')
      const cached = await cache?.match(request)
      if (!isGraph) return cached
      const response = cached ?? (await fetch(request))
      if (!response.ok)
        throw new Error(`Graph download failed (${response.status})`)
      const raw = await response.arrayBuffer()
      if (!cached && cache) {
        try {
          await cache.put(request, new Response(raw))
        } catch {
          /* Storage quota is optional. */
        }
      }
      status('Instrumenting Qwen MLP activations…')
      const patched = instrumentGraph(new Uint8Array(raw))
      return new Response(patched.slice().buffer as ArrayBuffer, {
        headers: { 'Content-Length': String(patched.byteLength) },
      })
    },
    async put(request: string, response: Response) {
      if (request.endsWith('.onnx')) return
      try {
        await cache?.put(request, response)
      } catch {
        /* Storage quota is optional. */
      }
    },
  }
}
