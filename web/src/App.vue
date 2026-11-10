<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import Composer from './components/Composer.vue'
import EngineRail from './components/EngineRail.vue'
import MessageView from './components/MessageView.vue'
import { useChat } from './composables/useChat'

const chat = useChat()
const { messages, status, phase, error, ctxUsed, ctxMax, trace, liveRate, settings } = chat

const draft = ref('')
const log = ref<HTMLElement | null>(null)
const composer = ref<InstanceType<typeof Composer> | null>(null)
const pinned = ref(true)

const reading = computed(() => [...messages.value].reverse().find((m) => m.role === 'user'))
const title = computed(() => status.value?.name ?? 'mia')

function onScroll() {
  const el = log.value
  if (el) pinned.value = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

// Follow the stream unless the reader scrolled up.
watch(
  () => messages.value.map((m) => m.content.length + (m.progress ?? 0)).join(),
  async () => {
    if (!pinned.value) return
    await nextTick()
    log.value?.scrollTo({ top: log.value.scrollHeight })
  },
)

async function send(text: string) {
  draft.value = ''
  pinned.value = true
  draft.value = await chat.send(text)
  composer.value?.focus()
}

async function reset() {
  await chat.reset()
  composer.value?.focus()
}

onMounted(async () => {
  await chat.load()
  composer.value?.focus()
})
</script>

<template>
  <div class="shell">
    <main class="conversation">
      <header class="top">
        <h1>{{ title }}</h1>
        <p v-if="status" class="sub">
          Running on this Mac with {{ status.model.active }} of {{ status.model.experts }} experts per token
        </p>
      </header>

      <section ref="log" class="log" aria-live="polite" @scroll="onScroll">
        <div v-if="!messages.length" class="empty">
          <p class="lead">Ask it anything.</p>
          <p>
            Replies come slowly, around one word per second. The model is 33 GB and most of it stays on disk, so every
            token loads the experts it needs. Watch the panel to see which ones were already in memory.
          </p>
        </div>
        <MessageView v-for="m in messages" :key="m.id" :message="m" />
      </section>

      <footer class="bottom">
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <Composer
          ref="composer"
          v-model="draft"
          :phase="phase"
          :disabled="!status"
          @send="send"
          @stop="chat.stop()"
        />
      </footer>
    </main>

    <EngineRail
      class="side"
      :status="status"
      :phase="phase"
      :ctx-used="ctxUsed"
      :ctx-max="ctxMax"
      :trace="trace"
      :live-rate="liveRate"
      :settings="settings"
      :reading="reading"
      @reset="reset"
    />
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  height: 100%;
}
.conversation {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-height: 0;
}
.top,
.log,
.bottom {
  padding-inline: max(1.5rem, calc((100% - 68ch) / 2));
}
.top {
  padding-block: 1.5rem 1rem;
  border-bottom: 1px solid var(--rule);
}
h1 {
  margin: 0;
  font-family: var(--serif);
  font-size: var(--step-2);
  font-weight: 500;
  letter-spacing: -0.01em;
}
.sub {
  margin: 0.25rem 0 0;
  color: var(--muted);
  font-size: var(--step--1);
}
.log {
  overflow-y: auto;
  padding-block: 0.5rem 2rem;
}
.empty {
  max-width: 52ch;
  margin-top: 18vh;
  color: var(--muted);
}
.empty .lead {
  margin: 0 0 0.75rem;
  font-family: var(--serif);
  font-size: var(--step-3);
  line-height: 1.15;
  color: var(--ink);
}
.bottom {
  padding-block: 0.75rem 1.25rem;
}
.error {
  margin: 0 0 0.625rem;
  padding: 0.625rem 0.875rem;
  border-left: 3px solid var(--danger);
  background: var(--paper-raised);
  color: var(--ink);
}
.side {
  background: var(--paper-raised);
  border-left: 1px solid var(--rule);
  overflow-y: auto;
}

@media (max-width: 860px) {
  .shell {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 1fr) auto;
    height: auto;
    min-height: 100%;
  }
  .conversation {
    height: 100dvh;
  }
  .top,
  .log,
  .bottom {
    padding-inline: 1rem;
  }
  .side {
    border-left: 0;
    border-top: 1px solid var(--rule);
  }
}
</style>
