<script setup>
import { useChatStore } from '../../store/chatStore'

defineProps({
  claims: {
    type: Array,
    required: true,
    default: () => []
  }
})

const store = useChatStore()
</script>

<template>
  <div class="traceable-paragraph">
    <span
      v-for="(claim, idx) in claims"
      :key="idx"
      class="claim-sentence"
      :class="{ 'direct-quote': claim.is_direct_quote }"
    >
      {{ claim.statement }}

      <sup
        v-if="claim.source_ids && claim.source_ids.length > 0"
        class="citation-badge"
        @click.stop="store.openEvidence(claim.source_ids)"
        title="Click to view original evidence"
      >
        [{{ claim.source_ids.length }} source]
      </sup>
    </span>
  </div>
</template>

<style scoped>
.traceable-paragraph {
  font-size: 1rem;
  line-height: 1.8;
  color: #333;
  text-align: justify;
}

.claim-sentence {
  display: inline;
  transition: background-color 0.2s;
}

/* 如果是直接截取原话，给一点极其微妙的背景色提示 */
.claim-sentence.direct-quote {
  background-color: rgba(64, 158, 255, 0.08);
  border-radius: 2px;
}

.citation-badge {
  color: #409EFF;
  cursor: pointer;
  font-size: 0.75rem;
  font-weight: 600;
  margin: 0 4px;
  padding: 2px 4px;
  background-color: #ecf5ff;
  border-radius: 4px;
  border: 1px solid #b3d8ff;
  user-select: none;
  transition: all 0.2s ease;
}

.citation-badge:hover {
  background-color: #409EFF;
  color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 2px 4px rgba(64, 158, 255, 0.3);
}
</style>