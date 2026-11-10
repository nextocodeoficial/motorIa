import { computed, reactive, ref } from 'vue'
import type { ChatClient, Stats, Status } from '../api/client'
import { httpClient } from '../api/client'

export interface UiMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  stats?: Stats
  /** User turns: share of the message the model has read (0..1). */
  progress?: number
  streaming?: boolean
}

export type Phase = 'idle' | 'reading' | 'writing'

export interface Settings {
  temperature: number
  topP: number
  maxTokens: number
  think: boolean
}

const SETTINGS_KEY = 'mia.settings'
const defaults: Settings = { temperature: 0.7, topP: 0.8, maxTokens: 1024, think: false }

function loadSettings(): Settings {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') }
  } catch {
    return { ...defaults }
  }
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s))
  } catch {
    /* private mode: settings just won't persist */
  }
}

let nextId = 1

export function useChat(client: ChatClient = httpClient) {
  const messages = ref<UiMessage[]>([])
  const status = ref<Status | null>(null)
  const phase = ref<Phase>('idle')
  const error = ref<string | null>(null)
  const ctxUsed = ref(0)
  const ctxMax = ref(0)
  /** Per generated token: share of expert loads served from RAM. */
  const trace = ref<number[]>([])
  const tokenTimes = ref<number[]>([])
  const settings = reactive<Settings>(loadSettings())

  const liveRate = computed(() => {
    const t = tokenTimes.value.slice(-12)
    if (t.length < 2) return 0
    return ((t.length - 1) * 1000) / (t[t.length - 1] - t[0])
  })

  async function load() {
    try {
      const [st, history] = await Promise.all([client.status(), client.messages()])
      status.value = st
      ctxUsed.value = st.ctx_used
      ctxMax.value = st.ctx_max
      messages.value = history.map((m) => ({ id: nextId++, ...m }))
    } catch (e) {
      error.value = `Can't reach the engine server: ${(e as Error).message}`
    }
  }

  /** Sends a message. Returns text to put back in the composer ('' if sent). */
  async function send(raw: string): Promise<string> {
    const text = raw.trim()
    if (!text || phase.value !== 'idle') return raw
    error.value = null
    trace.value = []
    tokenTimes.value = []

    const user = reactive<UiMessage>({ id: nextId++, role: 'user', content: text, progress: 0 })
    const answer = reactive<UiMessage>({ id: nextId++, role: 'assistant', content: '', streaming: true })
    messages.value.push(user, answer)
    phase.value = 'reading'

    let prev: { hits: number; misses: number } | null = null
    let failure: string | null = null
    try {
      for await (const ev of client.chat({
        message: text,
        temperature: settings.temperature,
        top_p: settings.topP,
        max_tokens: settings.maxTokens,
        think: settings.think,
      })) {
        switch (ev.type) {
          case 'progress':
            user.progress = ev.done / ev.total
            break
          case 'token': {
            phase.value = 'writing'
            user.progress = 1
            answer.content += ev.text
            tokenTimes.value.push(performance.now())
            if (prev) {
              const dh = ev.hits - prev.hits
              const dm = ev.misses - prev.misses
              if (dh + dm > 0) trace.value.push(dh / (dh + dm))
            }
            prev = { hits: ev.hits, misses: ev.misses }
            break
          }
          case 'done':
            answer.content = ev.message.content
            answer.stats = ev.message.stats
            ctxUsed.value = ev.ctx_used
            ctxMax.value = ev.ctx_max
            break
          case 'error':
            failure = ev.message
            break
        }
      }
    } catch (e) {
      failure = `Lost the connection to the engine server: ${(e as Error).message}`
    }

    answer.streaming = false
    phase.value = 'idle'
    if (failure) {
      messages.value = messages.value.filter((m) => m.id !== user.id && m.id !== answer.id)
      error.value = failure
      return text
    }
    return ''
  }

  async function stop() {
    await client.stop()
  }

  async function reset() {
    try {
      const st = await client.reset()
      messages.value = []
      trace.value = []
      ctxUsed.value = st.ctx_used
      ctxMax.value = st.ctx_max
      error.value = null
    } catch (e) {
      error.value = (e as Error).message
    }
  }

  return { messages, status, phase, error, ctxUsed, ctxMax, trace, liveRate, settings, load, send, stop, reset }
}
