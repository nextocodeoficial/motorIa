<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { Phase } from '../composables/useChat'

const props = defineProps<{ phase: Phase; disabled: boolean }>()
const emit = defineEmits<{ send: [text: string]; stop: [] }>()

const draft = defineModel<string>({ required: true })
const box = ref<HTMLTextAreaElement | null>(null)

function grow() {
  const el = box.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 240)}px`
}
watch(draft, () => nextTick(grow))

function onKey(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    submit()
  }
}

function submit() {
  if (props.phase === 'idle' && draft.value.trim() && !props.disabled) emit('send', draft.value)
}

defineExpose({ focus: () => box.value?.focus() })
</script>

<template>
  <form class="composer" @submit.prevent="submit">
    <label for="message" class="visually-hidden">Message</label>
    <textarea
      id="message"
      ref="box"
      v-model="draft"
      rows="1"
      placeholder="Write a message"
      :disabled="disabled"
      @keydown="onKey"
    />
    <button v-if="phase === 'idle'" type="submit" class="send" :disabled="disabled || !draft.trim()">Send</button>
    <button v-else type="button" class="stop" @click="emit('stop')">Stop</button>
  </form>
  <p class="hint">Enter sends, Shift+Enter adds a line</p>
</template>

<style scoped>
.composer {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  padding: 0.5rem;
  background: var(--paper-raised);
  border: 1px solid var(--rule);
  border-radius: 10px;
}
.composer:focus-within {
  border-color: var(--ram);
}
textarea {
  flex: 1;
  resize: none;
  border: 0;
  background: transparent;
  padding: 0.5rem;
  font-size: var(--step-1);
  line-height: 1.45;
  max-height: 240px;
}
textarea:focus {
  outline: none;
}
button {
  border: 0;
  border-radius: 7px;
  padding: 0.55rem 1.1rem;
  font-weight: 600;
  cursor: pointer;
}
.send {
  background: var(--ink);
  color: var(--paper);
}
.send:disabled {
  opacity: 0.35;
  cursor: default;
}
.stop {
  background: transparent;
  color: var(--ink);
  box-shadow: inset 0 0 0 1.5px var(--ink);
}
.hint {
  margin: 0.375rem 0.25rem 0;
  font-size: var(--step--1);
  color: var(--muted);
}
@media (hover: none) {
  .hint {
    display: none;
  }
}
</style>
