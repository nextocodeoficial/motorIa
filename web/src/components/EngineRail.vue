<script setup lang="ts">
import { computed, watch } from 'vue'
import type { Status } from '../api/client'
import type { Phase, Settings, UiMessage } from '../composables/useChat'
import { saveSettings } from '../composables/useChat'
import { formatNumber } from '../lib/markdown'
import ExpertTrace from './ExpertTrace.vue'

const props = defineProps<{
  status: Status | null
  phase: Phase
  ctxUsed: number
  ctxMax: number
  trace: number[]
  liveRate: number
  settings: Settings
  reading: UiMessage | undefined
}>()
const emit = defineEmits<{ reset: [] }>()

watch(() => ({ ...props.settings }), saveSettings, { deep: true })

const ctxShare = computed(() => (props.ctxMax ? props.ctxUsed / props.ctxMax : 0))

const activity = computed(() => {
  if (props.phase === 'reading') {
    const p = props.reading?.progress ?? 0
    return `Reading your message, ${Math.round(p * 100)}% done`
  }
  if (props.phase === 'writing') {
    return props.liveRate > 0 ? `Writing at ${props.liveRate.toFixed(1)} tokens per second` : 'Writing'
  }
  return 'Ready'
})
</script>

<template>
  <aside class="rail" aria-label="Engine">
    <section>
      <h2>Engine</h2>
      <p class="activity" :class="phase" role="status">{{ activity }}</p>
      <p v-if="status" class="fact">
        {{ formatNumber(status.model.slots) }} of {{ formatNumber(status.model.total_experts) }} experts fit in RAM
        ({{ status.model.cache_gib.toFixed(1) }} GiB). The rest are read from disk when a token needs them.
      </p>
    </section>

    <section>
      <h2>Where this reply's experts came from</h2>
      <ExpertTrace :ratios="trace" />
    </section>

    <section>
      <h2>Conversation</h2>
      <div
        class="meter"
        role="meter"
        :aria-valuenow="ctxUsed"
        aria-valuemin="0"
        :aria-valuemax="ctxMax"
        aria-label="Conversation length"
      >
        <span :style="{ width: `${ctxShare * 100}%` }" />
      </div>
      <p class="fact">{{ formatNumber(ctxUsed) }} of {{ formatNumber(ctxMax) }} tokens used</p>
      <button type="button" class="ghost" :disabled="phase !== 'idle'" @click="emit('reset')">New chat</button>
    </section>

    <section>
      <h2>Reply settings</h2>
      <label class="field">
        <span>Creativity <output>{{ settings.temperature.toFixed(1) }}</output></span>
        <input v-model.number="settings.temperature" type="range" min="0" max="1.5" step="0.1" />
        <small>0 always picks the most likely word</small>
      </label>
      <label class="field">
        <span>Longest reply <output>{{ settings.maxTokens }} tokens</output></span>
        <input v-model.number="settings.maxTokens" type="range" min="64" max="4096" step="64" />
      </label>
      <label class="check">
        <input v-model="settings.think" type="checkbox" />
        <span>Let the model reason before answering <small>Slower: it writes its reasoning first</small></span>
      </label>
    </section>
  </aside>
</template>

<style scoped>
.rail {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  padding: 1.5rem 1.25rem 2rem;
}
h2 {
  margin: 0 0 0.625rem;
  font-size: var(--step-0);
  font-weight: 600;
}
.activity {
  margin: 0 0 0.5rem;
  font-size: var(--step-1);
  font-variant-numeric: tabular-nums;
}
.activity.reading,
.activity.writing {
  color: var(--ram);
}
.fact {
  margin: 0.5rem 0 0;
  font-size: var(--step--1);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.meter {
  height: 6px;
  border-radius: 3px;
  background: var(--paper);
  overflow: hidden;
}
.meter span {
  display: block;
  height: 100%;
  background: var(--ink);
  transition: width 0.3s ease-out;
}
.ghost {
  margin-top: 0.875rem;
  background: none;
  border: 1px solid var(--rule);
  border-radius: 7px;
  padding: 0.45rem 0.9rem;
  cursor: pointer;
}
.ghost:hover:not(:disabled) {
  border-color: var(--ink);
}
.ghost:disabled {
  opacity: 0.4;
  cursor: default;
}
.field {
  display: grid;
  gap: 0.25rem;
  margin-bottom: 1rem;
}
.field span {
  display: flex;
  justify-content: space-between;
}
output {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
input[type='range'] {
  width: 100%;
  accent-color: var(--ram);
}
small {
  font-size: var(--step--1);
  color: var(--muted);
}
.check {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  cursor: pointer;
}
.check input {
  margin-top: 0.3rem;
  accent-color: var(--ram);
}
.check small {
  display: block;
}
</style>
