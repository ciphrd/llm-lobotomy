import { writeMask } from './mask'
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  Tensor,
  TextStreamer,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers'
import { configureCache } from './cache'
import {
  activationName,
  maskName,
  MODEL_ID,
  REVISION,
  LAYERS,
  CHANNELS,
} from './graph'
import { continuationLoss } from './metrics'
import { analyzeProfiles } from './statistics'
import {
  PROFILE_TEMPLATES,
  TEST_TEMPLATES,
  parseControls,
  renderProbe,
} from './probes'
import type {
  ActivationFrame,
  Message,
  Neuron,
  Report,
  Trial,
} from './protocol'

type Probe = {
  prompt: string
  group: 'target' | 'control'
  split: 'train' | 'test'
  ids: number[]
  promptLength: number
  baseline: number
  continuation: string
}

export class BrowserModel {
  private model!: PreTrainedModel
  private tokenizer!: PreTrainedTokenizer
  private masks: Record<string, Tensor> = {}
  private activeMask: Neuron[] = []
  private capture = false
  private activations: number[][] = []
  readonly stop = new InterruptableStoppingCriteria()
  private criteria = new StoppingCriteriaList()

  private step = 0

  private readonly onActivations?: (frame: ActivationFrame) => void

  constructor(onActivations?: (frame: ActivationFrame) => void) {
    this.onActivations = onActivations
    this.criteria.push(this.stop)
  }

  async load(status: (text: string) => void) {
    if (!navigator.gpu)
      throw new Error(
        'WebGPU is required. Use a compatible browser on localhost or HTTPS.',
      )
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter?.features.has('shader-f16'))
      throw new Error(
        'This Qwen ONNX export requires a GPU with shader-f16 support.',
      )
    await configureCache(status)
    this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      revision: REVISION,
    })
    this.model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      revision: REVISION,
      device: 'webgpu',
      dtype: 'q4f16',
      use_external_data_format: 2,
      progress_callback: (p) => {
        if (p.status === 'progress')
          status(`Loading ${p.file} · ${Math.round(p.progress)}%`)
        else if (p.status === 'done') status('Preparing GPU…')
      },
    })
    for (let layer = 0; layer < LAYERS; layer++)
      this.masks[maskName(layer)] = new Tensor(
        'float16',
        new Uint16Array(CHANNELS).fill(0x3c00),
        [CHANNELS],
      )
    const original = this.model.forward.bind(this.model)
    this.model.forward = async (inputs: Record<string, unknown>) => {
      const outputs: Record<string, Tensor> = await original({
        ...inputs,
        ...this.masks,
      })
      const values = this.onActivations
        ? new Float32Array(LAYERS * CHANNELS)
        : null
      for (let layer = 0; layer < LAYERS; layer++) {
        const name = activationName(layer),
          summary = outputs[name]
        if (!summary)
          throw new Error(
            'Instrumented activation output missing; refusing to report synthetic scores',
          )
        if (this.capture)
          this.activations[layer] = Array.from(summary.data as Float32Array)
        if (values) {
          values.set(summary.data as Float32Array, layer * CHANNELS)
          const mask = this.masks[maskName(layer)].data as Uint16Array
          for (let channel = 0; channel < CHANNELS; channel++)
            if (mask[channel] === 0) values[layer * CHANNELS + channel] = 0
        }
        summary.dispose()
        delete outputs[name]
      }
      if (values)
        this.onActivations?.({
          values,
          step: ++this.step,
          tokens: (inputs.input_ids as Tensor).dims.at(-1) ?? 1,
        })
      // Yield a task so Stop messages are handled between native WebGPU steps.
      await new Promise<void>((resolve) => setTimeout(resolve, 0))
      return outputs
    }
  }

  setMask(neurons: Neuron[]) {
    if (
      neurons.some(
        (n) =>
          !Number.isInteger(n.layer) ||
          !Number.isInteger(n.channel) ||
          n.layer < 0 ||
          n.layer >= LAYERS ||
          n.channel < 0 ||
          n.channel >= CHANNELS,
      )
    )
      throw new Error('Invalid neuron mask')
    this.activeMask = neurons
    for (let layer = 0; layer < LAYERS; layer++) {
      writeMask(
        this.masks[maskName(layer)].data as Uint16Array,
        CHANNELS,
        neurons.filter((n) => n.layer === layer).map((n) => n.channel),
      )
    }
  }

  private checkStop() {
    if (this.stop.interrupted)
      throw new Error('Experiment stopped. The prior mask is restored.')
  }
  private inputs(ids: number[]) {
    return {
      input_ids: new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [
        1,
        ids.length,
      ]),
      attention_mask: new Tensor(
        'int64',
        new BigInt64Array(ids.length).fill(1n),
        [1, ids.length],
      ),
    }
  }

  async generate(messages: Message[], onText: (text: string) => void) {
    this.stop.reset()
    const options = {
      tokenize: false as const,
      add_generation_prompt: true,
      enable_thinking: false,
    }
    const text = this.tokenizer.apply_chat_template(messages, options)
    const ids = this.tokenizer.encode(text, { add_special_tokens: false })
    if (ids.length > 3072)
      throw new Error(
        'Conversation too long. Clear the conversation and try again.',
      )
    const inputs = this.inputs(ids)
    try {
      const output = await this.model.generate({
        ...inputs,
        max_new_tokens: 512,
        do_sample: false,
        stopping_criteria: this.criteria,
        streamer: new TextStreamer(this.tokenizer, {
          skip_prompt: true,
          callback_function: onText,
        }),
      })
      if (output instanceof Tensor) output.dispose()
    } finally {
      inputs.input_ids.dispose()
      inputs.attention_mask.dispose()
    }
  }

  private async baseline(
    prompt: string,
    group: Probe['group'],
    split: Probe['split'],
  ): Promise<Probe> {
    this.checkStop()
    const ids = this.tokenizer.encode(prompt, { add_special_tokens: false })
    const inputs = this.inputs(ids)
    try {
      const output = await this.model.generate({
        ...inputs,
        max_new_tokens: 12,
        do_sample: false,
        stopping_criteria: this.criteria,
      })
      if (!(output instanceof Tensor))
        throw new Error('Unexpected generation output')
      const all = Array.from(output.data as BigInt64Array, Number)
      output.dispose()
      const continuation = this.tokenizer.decode(all.slice(ids.length), {
        skip_special_tokens: true,
      })
      return {
        prompt,
        group,
        split,
        ids: all,
        promptLength: ids.length,
        baseline: 0,
        continuation,
      }
    } finally {
      inputs.input_ids.dispose()
      inputs.attention_mask.dispose()
    }
  }

  private async score(probe: Probe, capture = false): Promise<number> {
    this.checkStop()
    this.capture = capture
    const inputs = this.inputs(probe.ids)
    let output: Record<string, Tensor> | undefined
    try {
      output = await this.model.forward(inputs)
      const logits = output!.logits
      if (
        logits.dims[1] !== probe.ids.length ||
        logits.data.length !== probe.ids.length * logits.dims[2]
      )
        throw new Error(
          `Scoring requires full logits: got ${logits.type} [${logits.dims}] for ${probe.ids.length} tokens; data length ${logits.data.length}`,
        )
      return continuationLoss(
        logits.data as Float32Array,
        logits.dims[2],
        probe.ids,
        probe.promptLength,
      )
    } finally {
      this.capture = false
      for (const t of Object.values(output ?? {}))
        if (t instanceof Tensor) t.dispose()
      inputs.input_ids.dispose()
      inputs.attention_mask.dispose()
    }
  }

  private async profile(prompt: string): Promise<Float32Array> {
    this.checkStop()
    const inputs = this.inputs(
      this.tokenizer.encode(prompt, { add_special_tokens: false }),
    )
    this.capture = true
    let output: Record<string, Tensor> | undefined
    try {
      output = await this.model.forward(inputs)
      const values = new Float32Array(LAYERS * CHANNELS)
      for (let layer = 0; layer < LAYERS; layer++)
        values.set(this.activations[layer], layer * CHANNELS)
      this.checkStop()
      return values
    } finally {
      this.capture = false
      for (const t of Object.values(output ?? {}))
        if (t instanceof Tensor) t.dispose()
      inputs.input_ids.dispose()
      inputs.attention_mask.dispose()
    }
  }

  async analyze(
    concept: string,
    status: (text: string) => void,
  ): Promise<Report> {
    this.stop.reset()
    this.setMask([])
    let controls: string[] = [],
      controlGeneration = ''
    try {
      status('Generating related control words…')
      for (let attempt = 0; attempt < 2; attempt++) {
        this.checkStop()
        const text = this.tokenizer.apply_chat_template(
          [
            {
              role: 'user',
              content: `For the concept ${JSON.stringify(concept)}, choose exactly three related but distinct contrast concepts. Choose concrete sibling alternatives at the SAME level of specificity. Examples: apple -> ["pear", "peach", "banana"]; violin -> ["cello", "flute", "trumpet"]; wolf -> ["fox", "dog", "coyote"]. For a specific animal, choose three other specific animals. Never use umbrella categories such as "predator", "mammal", or "domestic animal". Never use synonyms (for example cat/feline), translations, attributes, subtypes, or singular/plural variants. Prefer single words for a single-word target. Return ONLY a JSON array of three short strings.${attempt ? ' The previous response was invalid; follow the JSON format exactly.' : ''}`,
            },
          ],
          {
            tokenize: false,
            add_generation_prompt: true,
            ...{ enable_thinking: false },
          },
        )
        const ids = this.tokenizer.encode(text, { add_special_tokens: false }),
          inputs = this.inputs(ids)
        try {
          const output = await this.model.generate({
            ...inputs,
            max_new_tokens: 128,
            do_sample: false,
            stopping_criteria: this.criteria,
          })
          if (!(output instanceof Tensor))
            throw new Error('Unexpected control-generation output')
          try {
            controlGeneration = this.tokenizer.decode(
              Array.from(output.data as BigInt64Array).slice(ids.length),
              { skip_special_tokens: true },
            )
          } finally {
            output.dispose()
          }
          this.checkStop()
          try {
            controls = parseControls(controlGeneration, concept)
            break
          } catch (error) {
            if (attempt === 1)
              throw new Error(
                `Could not generate valid controls: ${error instanceof Error ? error.message : String(error)}`,
              )
          }
        } finally {
          inputs.input_ids.dispose()
          inputs.attention_mask.dispose()
        }
      }
      const targets: Float32Array[] = [],
        comparisons: Float32Array[] = []
      for (let i = 0; i < PROFILE_TEMPLATES.length; i++) {
        status(
          `Profiling ${i + 1}/${PROFILE_TEMPLATES.length} · controls: ${controls.join(', ')}`,
        )
        const template = PROFILE_TEMPLATES[i]
        targets.push(await this.profile(renderProbe(template, concept)))
        const average = new Float32Array(LAYERS * CHANNELS)
        for (const word of controls) {
          const values = await this.profile(renderProbe(template, word))
          for (let j = 0; j < average.length; j++)
            average[j] += values[j] / controls.length
        }
        comparisons.push(average)
      }
      status('Computing paired statistics and multiple-comparison correction…')
      const { candidates: ranked, ...statistics } = analyzeProfiles(
        targets,
        comparisons,
        CHANNELS,
      )
      targets.length = 0
      comparisons.length = 0
      return {
        concept,
        controls,
        controlGeneration,
        statistics,
        candidates: ranked,
        probes: PROFILE_TEMPLATES.flatMap((t) =>
          [concept, ...controls].map((word, i) => ({
            prompt: renderProbe(t, word),
            continuation: '',
            group: i ? 'control' : 'target',
            split: 'profile',
          })),
        ),
      }
    } finally {
      this.setMask([])
    }
  }

  async evaluate(
    report: Report,
    neurons: Neuron[],
    status: (text: string) => void,
  ): Promise<Trial> {
    const previousMask = this.activeMask
    this.stop.reset()
    try {
      this.setMask([])
      const probes: Probe[] = []
      for (const template of TEST_TEMPLATES) {
        for (const word of [report.concept, ...report.controls]) {
          status(
            `Baseline ${probes.length + 1}/${TEST_TEMPLATES.length * (report.controls.length + 1)} · ${word}`,
          )
          const probe = await this.baseline(
            renderProbe(template, word),
            word === report.concept ? 'target' : 'control',
            'test',
          )
          probe.baseline = await this.score(probe)
          probes.push(probe)
        }
      }
      this.setMask(neurons)
      const differences = { target: [] as number[], control: [] as number[] }
      const samples: Trial['samples'] = []
      for (const [i, probe] of probes.entries()) {
        status(`Testing threshold mask · ${i + 1}/${probes.length}`)
        const delta = (await this.score(probe)) - probe.baseline
        if (!Number.isFinite(delta))
          throw new Error('Non-finite evaluation loss')
        differences[probe.group].push(delta)
        samples.push({
          prompt: probe.prompt,
          group: probe.group,
          continuation: probe.continuation,
          delta,
        })
      }
      this.checkStop()
      const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
      return {
        samples,
        count: neurons.length,
        targetDelta: mean(differences.target),
        controlDelta: mean(differences.control),
      }
    } finally {
      this.setMask(previousMask)
    }
  }

  async dispose() {
    for (const mask of Object.values(this.masks)) mask.dispose()
    await this.model?.dispose()
  }
}
