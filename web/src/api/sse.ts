export interface SSEMessage {
  event: string
  data: string
}

/**
 * Parses a text/event-stream body. Used instead of EventSource because the
 * chat endpoint is a POST.
 */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = body.getReader()
  const decoder = new TextDecoder() // stream mode keeps split UTF-8 runes intact
  let buf = ''
  let event = 'message'
  let data: string[] = []
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '')
        buf = buf.slice(nl + 1)
        if (line === '') {
          if (data.length) yield { event, data: data.join('\n') }
          event = 'message'
          data = []
        } else if (line.startsWith('event:')) {
          event = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          data.push(line.slice(5).replace(/^ /, ''))
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
