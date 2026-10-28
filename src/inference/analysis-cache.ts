import type { Report, CachedAnalysisOption } from './protocol'

// Bump whenever prompts, control generation, activation collection, scoring,
// statistical tests, or the report schema change. Model identity is added by the caller.
export const ANALYSIS_METHOD_VERSION = '3'
const PREFIX = 'qwen-analysis-'
export type CachedAnalysis = { report: Report; savedAt: string }

/** Independent from the multi-GB model-download cache. Storage is best effort. */
export class AnalysisCache {
  private cache: Cache | null = null
  private initialized = false
  private readonly version: string
  private readonly storage: CacheStorage
  constructor(version: string, storage: CacheStorage) {
    this.version = version
    this.storage = storage
  }

  async initialize() {
    if (this.initialized) return
    this.initialized = true
    try {
      const name = PREFIX + encodeURIComponent(this.version)
      for (const key of await this.storage.keys()) {
        if (key.startsWith(PREFIX) && key !== name)
          await this.storage.delete(key)
      }
      this.cache = await this.storage.open(name)
    } catch {
      this.cache = null
    }
  }

  private key(term: string, controls?: string[]) {
    // Preserve case: US/us and proper names can denote different concepts.
    return `https://analysis-cache.invalid/term/${encodeURIComponent(JSON.stringify([term.trim(), controls?.map((s) => s.trim()) ?? null]))}`
  }

  async get(term: string, controls?: string[]): Promise<CachedAnalysis | null> {
    await this.initialize()
    try {
      const response = await this.cache?.match(this.key(term, controls))
      if (!response) return null
      const entry = await response.json()
      if (
        entry.version !== this.version ||
        entry.report?.concept !== term.trim() ||
        !Array.isArray(entry.report?.candidates) ||
        !Array.isArray(entry.report?.probes) ||
        !Array.isArray(entry.report?.controls) ||
        !entry.report?.statistics ||
        entry.report.controlSource !== (controls ? 'provided' : 'generated') ||
        (controls &&
          JSON.stringify(entry.report.controls) !== JSON.stringify(controls)) ||
        typeof entry.savedAt !== 'string'
      ) {
        await this.cache?.delete(this.key(term, controls))
        return null
      }
      return { report: entry.report, savedAt: entry.savedAt }
    } catch {
      return null
    }
  }

  async list(): Promise<CachedAnalysisOption[]> {
    await this.initialize()
    if (!this.cache) return []
    const entries: CachedAnalysisOption[] = []
    try {
      for (const request of await this.cache.keys()) {
        try {
          const response = await this.cache.match(request)
          if (!response) continue
          const header = response.headers.get('X-Analysis-Summary')
          let summary: CachedAnalysisOption
          if (header) summary = JSON.parse(decodeURIComponent(header))
          else {
            // Existing v3 entries predate the lightweight listing header.
            const entry = await response.json()
            if (entry.version !== this.version) continue
            summary = {
              concept: entry.report.concept,
              controls: entry.report.controls,
              controlSource: entry.report.controlSource,
              savedAt: entry.savedAt,
            }
          }
          if (
            typeof summary.concept === 'string' &&
            Array.isArray(summary.controls) &&
            summary.controls.length === 3 &&
            summary.controls.every((s) => typeof s === 'string') &&
            ['generated', 'provided'].includes(summary.controlSource) &&
            typeof summary.savedAt === 'string'
          )
            entries.push(summary)
        } catch {
          /* Ignore individual malformed entries. */
        }
      }
    } catch {
      /* Storage is optional. */
    }
    return entries.sort(
      (a, b) =>
        b.savedAt.localeCompare(a.savedAt) ||
        a.concept.localeCompare(b.concept),
    )
  }

  async put(report: Report): Promise<string | null> {
    await this.initialize()
    if (!this.cache) return null
    const savedAt = new Date().toISOString()
    try {
      await this.cache.put(
        this.key(
          report.concept,
          report.controlSource === 'provided' ? report.controls : undefined,
        ),
        new Response(
          JSON.stringify({ version: this.version, savedAt, report }),
          {
            headers: {
              'Content-Type': 'application/json',
              'X-Analysis-Summary': encodeURIComponent(
                JSON.stringify({
                  concept: report.concept,
                  controls: report.controls,
                  controlSource: report.controlSource,
                  savedAt,
                }),
              ),
            },
          },
        ),
      )
      return savedAt
    } catch {
      return null
    }
  }
}
