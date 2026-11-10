export interface ThinkParts {
  /** Qwen3 reasoning text, or null when the reply has none. */
  reasoning: string | null
  answer: string
  /** The reasoning block is still being written. */
  thinking: boolean
}

const OPEN = '<think>'
const CLOSE = '</think>'

export function splitThink(text: string): ThinkParts {
  const start = text.trimStart()
  if (!start.startsWith(OPEN)) return { reasoning: null, answer: text, thinking: false }
  const body = start.slice(OPEN.length)
  const end = body.indexOf(CLOSE)
  if (end < 0) return { reasoning: body.trim(), answer: '', thinking: true }
  const reasoning = body.slice(0, end).trim()
  return {
    reasoning: reasoning === '' ? null : reasoning,
    answer: body.slice(end + CLOSE.length).trimStart(),
    thinking: false,
  }
}
