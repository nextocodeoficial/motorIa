import { describe, expect, it } from 'vitest'
import { parseSSE } from './sse'

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch))
      c.close()
    },
  })
}

async function collect(s: ReadableStream<Uint8Array>) {
  const out = []
  for await (const e of parseSSE(s)) out.push(e)
  return out
}

describe('parseSSE', () => {
  it('parses named events with JSON data', async () => {
    const evs = await collect(streamOf('event: token\ndata: {"text":"hi"}\n\nevent: done\ndata: {}\n\n'))
    expect(evs).toEqual([
      { event: 'token', data: '{"text":"hi"}' },
      { event: 'done', data: '{}' },
    ])
  })

  it('reassembles events split across chunks, including inside UTF-8', async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"text":"¡ñ"}\n\n')
    const cut = bytes.indexOf(0xc3) + 1 // split in the middle of "ñ"
    const s = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, cut))
        c.enqueue(bytes.slice(cut))
        c.close()
      },
    })
    const evs = await collect(s)
    expect(JSON.parse(evs[0].data).text).toBe('¡ñ')
  })

  it('accepts CRLF line endings and defaults the event name to message', async () => {
    const evs = await collect(streamOf('data: 1\r\n\r\n'))
    expect(evs).toEqual([{ event: 'message', data: '1' }])
  })
})
