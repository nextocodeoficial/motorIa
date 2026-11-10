<script setup lang="ts">
import { computed } from 'vue'

/** Per token: share of expert loads served from RAM (0..1). */
const props = defineProps<{ ratios: number[] }>()

const SLOTS = 96
const H = 48

const bars = computed(() => {
  const recent = props.ratios.slice(-SLOTS)
  return recent.map((r, i) => ({ x: i + (SLOTS - recent.length), ram: r * H }))
})

const average = computed(() => {
  if (!props.ratios.length) return null
  return props.ratios.reduce((a, b) => a + b, 0) / props.ratios.length
})
</script>

<template>
  <figure class="trace">
    <svg
      :viewBox="`0 0 ${SLOTS} ${H}`"
      preserveAspectRatio="none"
      role="img"
      :aria-label="
        average === null
          ? 'No tokens generated yet'
          : `${Math.round(average * 100)}% of expert loads came from RAM in this reply`
      "
    >
      <rect class="track" x="0" y="0" :width="SLOTS" :height="H" />
      <g v-for="b in bars" :key="b.x">
        <rect class="disk" :x="b.x + 0.12" y="0" width="0.76" :height="H - b.ram" />
        <rect class="ram" :x="b.x + 0.12" :y="H - b.ram" width="0.76" :height="b.ram" />
      </g>
    </svg>
    <figcaption>
      <span class="key"><i class="swatch ram" />from RAM</span>
      <span class="key"><i class="swatch disk" />read from disk</span>
      <span v-if="average !== null" class="avg">{{ Math.round(average * 100) }}% from RAM</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.trace {
  margin: 0;
}
svg {
  display: block;
  width: 100%;
  height: 72px;
}
.track {
  fill: var(--paper);
}
.ram {
  fill: var(--ram);
}
.disk {
  fill: var(--disk);
}
figcaption {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 0.875rem;
  margin-top: 0.5rem;
  font-size: var(--step--1);
  color: var(--muted);
}
.key {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
}
.swatch {
  width: 0.625rem;
  height: 0.625rem;
  border-radius: 2px;
}
.swatch.ram {
  background: var(--ram);
}
.swatch.disk {
  background: var(--disk);
}
.avg {
  margin-left: auto;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
</style>
