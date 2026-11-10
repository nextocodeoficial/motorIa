import { parseSSE } from './sse'

export interface ModelInfo {
  layers: number
  experts: number
  active: number
  ctx: number
  slots: number
  total_experts: number
  wtype: string
  dense_gib: number
  cache_gib: number
}

export interface Status {
  name: string
  model: ModelInfo
  ctx_used: number
  ctx_max: number
  busy: boolean
}

export interface Stats {
  prompt_tokens: number
  gen_tokens: number
  prefill_sec: number
  gen_sec: number
  tokens_per_sec: number
  hit_rate: number
  reason: 'stop' | 'length' | 'ctx' | 'cancel'
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  stats?: Stats
}

export interface ChatRequest {
  message: string
  temperature: number
  top_p: number
  max_tokens: number
  think: boolean
}

export type ChatEvent =
  | { type: 'progress'; done: number; total: number }
  | { type: 'token'; text: string; hits: number; misses: number }
  | { type: 'done'; message: Message; ctx_used: number; ctx_max: number }
  | { type: 'error'; code: string; message: string }

export interface ChatClient {
  status(): Promise<Status>
  messages(): Promise<Message[]>
  reset(): Promise<Status>
  stop(): Promise<void>
  chat(req: ChatRequest, signal?: AbortSignal): AsyncIterable<ChatEvent>
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error ?? `request failed with status ${res.status}`)
  return body as T
}

export const httpClient: ChatClient = {
  status: () => fetch('/api/status').then((r) => json<Status>(r)),
  messages: () => fetch('/api/messages').then((r) => json<Message[]>(r)),
  reset: () => fetch('/api/reset', { method: 'POST' }).then((r) => json<Status>(r)),
  stop: async () => {
    await fetch('/api/stop', { method: 'POST' })
  },
  async *chat(req, signal) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal,
    })
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({}))
      yield { type: 'error', code: String(res.status), message: body.error ?? 'the server rejected the message' }
      return
    }
    for await (const m of parseSSE(res.body)) {
      yield { type: m.event, ...JSON.parse(m.data) } as ChatEvent
    }
  },
}
