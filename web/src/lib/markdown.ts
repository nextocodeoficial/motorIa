import DOMPurify from 'dompurify'
import { marked } from 'marked'

marked.setOptions({ gfm: true, breaks: false })

/** Model output is untrusted: always sanitize before v-html. */
export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(marked.parse(text, { async: false }))
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat('en-US').format(n)
}
