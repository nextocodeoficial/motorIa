import { describe, expect, it } from 'vitest'
import { splitThink } from './think'

describe('splitThink', () => {
  it('passes plain answers through', () => {
    expect(splitThink('Paris.')).toEqual({ reasoning: null, answer: 'Paris.', thinking: false })
  })

  it('separates a finished reasoning block', () => {
    expect(splitThink('<think>\nFrance → Paris\n</think>\n\nParis.')).toEqual({
      reasoning: 'France → Paris',
      answer: 'Paris.',
      thinking: false,
    })
  })

  it('reports a reasoning block that is still streaming', () => {
    expect(splitThink('<think>\nLet me')).toEqual({ reasoning: 'Let me', answer: '', thinking: true })
  })

  it('treats an empty block as no reasoning', () => {
    expect(splitThink('<think>\n\n</think>\n\nHi')).toEqual({ reasoning: null, answer: 'Hi', thinking: false })
  })
})
