<script setup lang="ts">
import { computed } from 'vue'
import type { UiMessage } from '../composables/useChat'
import { renderMarkdown } from '../lib/markdown'
import { splitThink } from '../lib/think'

const props = defineProps<{ message: UiMessage }>()

const parts = computed(() => splitThink(props.message.content))
const html = computed(() => renderMarkdown(parts.value.answer))

const reading = computed(() => props.message.role === 'user' && (props.message.progress ?? 1) < 1)

const summary = computed(() => {
  const s = props.message.stats
  if (!s) return ''
  const pieces = [`${s.gen_tokens} tokens in ${Math.round(s.gen_sec)} s`]
  if (s.gen_tokens > 0) pieces.push(`${s.tokens_per_sec.toFixed(1)} per second`)
  pieces.push(`${Math.round(s.hit_rate * 100)}% of experts from RAM`)
  return pieces.join(', ')
})

const ending = computed(() => {
  switch (props.message.stats?.reason) {
    case 'cancel':
      return 'Stopped'
    case 'length':
      return 'Reached the reply length limit'
    case 'ctx':
      return 'The conversation is full. Start a new chat to keep going.'
    default:
      return ''
  }
})
</script>

<template>
  <article v-if="message.role === 'user'" class="user" :aria-busy="reading">
    <p class="user-text" :style="{ '--p': message.progress ?? 1 }">{{ message.content }}</p>
  </article>

  <article v-else class="assistant" :aria-busy="message.streaming">
    <details v-if="parts.reasoning !== null" class="reasoning" :open="parts.thinking">
      <summary>{{ parts.thinking ? 'Reasoning…' : 'Reasoning' }}</summary>
      <p>{{ parts.reasoning }}</p>
    </details>
    <div class="prose" :class="{ streaming: message.streaming }" v-html="html" />
    <p v-if="ending" class="ending">{{ ending }}</p>
    <p v-if="summary" class="stats">{{ summary }}</p>
  </article>
</template>

<style scoped>
.user {
  margin: 2.5rem 0 1rem;
}
.user-text {
  margin: 0;
  max-width: 62ch;
  font-size: var(--step-1);
  line-height: 1.55;
  white-space: pre-wrap;
  /* Underline that fills in reading order while the model reads the message. */
  display: inline;
  background-image: linear-gradient(var(--ram), var(--ram));
  background-repeat: no-repeat;
  background-position: 0 100%;
  background-size: calc(var(--p) * 100%) 2px;
  padding-bottom: 3px;
  transition: background-size 0.4s ease-out;
}

.assistant {
  max-width: 68ch;
  font-family: var(--serif);
  font-size: 1.1875rem;
  line-height: 1.65;
}
.prose :deep(p) {
  margin: 0 0 1em;
}
.prose :deep(h1),
.prose :deep(h2),
.prose :deep(h3) {
  font-family: var(--sans);
  font-size: var(--step-1);
  font-weight: 600;
  margin: 1.5em 0 0.5em;
}
.prose :deep(ul),
.prose :deep(ol) {
  padding-left: 1.25em;
  margin: 0 0 1em;
}
.prose :deep(code) {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.82em;
  background: var(--paper-raised);
  padding: 0.1em 0.3em;
  border-radius: 3px;
}
.prose :deep(pre) {
  background: var(--paper-raised);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 0.875rem 1rem;
  overflow-x: auto;
  line-height: 1.45;
}
.prose :deep(pre code) {
  background: none;
  padding: 0;
}
.prose :deep(blockquote) {
  margin: 0 0 1em;
  padding-left: 1em;
  border-left: 2px solid var(--rule);
  color: var(--muted);
}
.prose :deep(:last-child) {
  margin-bottom: 0;
}

.reasoning {
  margin-bottom: 1rem;
  font-family: var(--sans);
  font-size: var(--step-0);
  color: var(--muted);
}
.reasoning summary {
  cursor: pointer;
  width: fit-content;
}
.reasoning p {
  margin: 0.5rem 0 0;
  padding-left: 0.875rem;
  border-left: 2px solid var(--rule);
  white-space: pre-wrap;
  font-style: italic;
  font-family: var(--serif);
}

.ending,
.stats {
  font-family: var(--sans);
  font-size: var(--step--1);
  color: var(--muted);
  margin: 0.75rem 0 0;
}
.ending {
  color: var(--ink);
}
.stats {
  font-variant-numeric: tabular-nums;
}
</style>
