import { describe, expect, it, vi } from 'vitest'
import type { ChatClient, ChatEvent, Status } from '../api/client'
import { useChat } from './useChat'

const status: Status = {
  name: 'qwen3-30b-a3b-q8',
  model: { layers: 48, experts: 128, active: 8, ctx: 4096, slots: 396, total_experts: 6144, wtype: 'q8_0', dense_gib: 1.65, cache_gib: 1.96 },
  ctx_used: 0,
  ctx_max: 4096,
  busy: false,
}

function fakeClient(events: ChatEvent[]): ChatClient {
  return {
    status: vi.fn(async () => status),
    messages: vi.fn(async () => []),
    reset: vi.fn(async () => status),
    stop: vi.fn(async () => {}),
    async *chat() {
      for (const e of events) yield e
    },
  }
}

describe('useChat', () => {
  it('streams a reply, tracks reading progress and the expert trace', async () => {
    const client = fakeClient([
      { type: 'progress', done: 1, total: 2 },
      { type: 'progress', done: 2, total: 2 },
      { type: 'token', text: 'Hola', hits: 10, misses: 10 },
      { type: 'token', text: ' mundo', hits: 40, misses: 20 },
      {
        type: 'done',
        message: { role: 'assistant', content: 'Hola mundo', stats: { prompt_tokens: 2, gen_tokens: 2, prefill_sec: 1, gen_sec: 2, tokens_per_sec: 1, hit_rate: 0.66, reason: 'stop' } },
        ctx_used: 12,
        ctx_max: 4096,
      },
    ])
    const chat = useChat(client)
    await chat.load()
    await chat.send('  hola  ')

    const [user, answer] = chat.messages.value
    expect(user).toMatchObject({ role: 'user', content: 'hola', progress: 1 })
    expect(answer).toMatchObject({ role: 'assistant', content: 'Hola mundo' })
    expect(answer.stats?.reason).toBe('stop')
    // Second token: 30 more hits, 10 more misses -> 75% from RAM.
    expect(chat.trace.value).toEqual([0.75])
    expect(chat.ctxUsed.value).toBe(12)
    expect(chat.phase.value).toBe('idle')
  })

  it('removes the turn and restores the draft when the engine rejects it', async () => {
    const chat = useChat(fakeClient([{ type: 'error', code: 'context_full', message: 'only 3 tokens left; start a new chat' }]))
    const draft = await chat.send('una pregunta larga')
    expect(chat.messages.value).toEqual([])
    expect(chat.error.value).toContain('start a new chat')
    expect(draft).toBe('una pregunta larga')
  })

  it('ignores blank input', async () => {
    const client = fakeClient([])
    const chat = useChat(client)
    await chat.send('   ')
    expect(chat.messages.value).toEqual([])
  })

  it('stop asks the server to cancel instead of dropping the stream', async () => {
    const client = fakeClient([])
    const chat = useChat(client)
    await chat.stop()
    expect(client.stop).toHaveBeenCalledOnce()
  })

  it('reset clears the conversation', async () => {
    const client = fakeClient([
      { type: 'token', text: 'x', hits: 1, misses: 1 },
      { type: 'done', message: { role: 'assistant', content: 'x' }, ctx_used: 5, ctx_max: 4096 },
    ])
    const chat = useChat(client)
    await chat.send('hola')
    await chat.reset()
    expect(chat.messages.value).toEqual([])
    expect(chat.ctxUsed.value).toBe(0)
  })
})
