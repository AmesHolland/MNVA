<script setup>
import { computed } from 'vue'
import { useChatStore } from '../../store/chatStore'

defineProps({
  claims: {
    type: Array,
    required: true,
    default: () => []
  }
})

const store = useChatStore()

// 计算这句 Claim 是否应该变暗 (Dimmed)
const isClaimDimmed = (claim) => {
  const timeRange = store.brushState?.timeRange
  const activeLabels = store.brushState?.spatialLabels || []

  // 如果没有任何刷选，全部点亮
  if (!timeRange && activeLabels.length === 0) return false

  // AI 主观洞察 (没有源新闻)，在刷选时默认不让它变暗，因为它属于总结性全局观点
  if (claim.is_subjective_insight) return false;

  // 如果客观事实没有来源，直接变暗
  if (!claim.source_ids || claim.source_ids.length === 0) return true

  const pool = store.analysisResults?.evidence_pool || []
  const [brushStart, brushEnd] = timeRange || [0, Infinity]

  const hasMatch = claim.source_ids.some(docId => {
    if (!docId) return false;
    const doc = pool.find(item => item?.DOC_ID === docId);
    if (!doc) return false

    let timeMatch = true
    if (doc.publish_date) {
      const docTime = new Date(doc.publish_date).getTime()
      timeMatch = docTime >= brushStart && docTime <= brushEnd
    }

    return timeMatch
  })

  return !hasMatch
}
</script>

<template>
  <div class="traceable-paragraph">
    <span
      v-for="(claim, idx) in claims"
      :key="idx"
      class="claim-sentence"
      :class="{
        'is-insight': claim.is_subjective_insight,
        'is-dimmed': isClaimDimmed(claim)
      }"
    >
      <span class="claim-text">{{ claim.content || claim.statement }}</span>

      <sup
        v-if="!claim.is_subjective_insight && claim.source_ids && claim.source_ids.length > 0"
        class="citation-badge"
        @click.stop="store.openEvidence(
          claim.source_ids,
          claim.claim_id || claim.content || claim.statement,
          claim.content || claim.statement
        )"
        title="Click to view the original evidence"
      >
        [{{ claim.source_ids.length }} source]
      </sup>

      <sup
        v-if="claim.is_subjective_insight"
        class="insight-badge"
        @click.stop="store.openEvidence(
          claim.source_ids,
          claim.claim_id || claim.content || claim.statement,
          claim.content || claim.statement
        )"
        title="AI Strategic Deduction and Insights"
      >
        💡 AI Insight [{{ claim.source_ids.length }}]
      </sup>

      <span class="sentence-spacer"> </span>
    </span>
  </div>
</template>

<style scoped>
.traceable-paragraph {
  font-size: 14.5px;
  line-height: 1.6; /* 增加行高，提升大段文字的阅读舒适度 */
  color: #333;
  text-align: justify; /* 两端对齐，报告显得更严谨 */
  margin-bottom: 12px;
}

/* 恢复为行内元素 */
.claim-sentence {
  display: inline;
  transition: opacity 0.4s ease, color 0.4s ease;
}

/* 褪色退居背景的样式 (刷选时生效) */
.claim-sentence.is-dimmed {
  opacity: 0.25;
  color: #909399;
}

.claim-sentence.is-dimmed .citation-badge,
.claim-sentence.is-dimmed .insight-badge {
  background-color: transparent;
  border-color: transparent;
  color: #c0c4cc;
  opacity: 0.5;
}

/* AI 洞察的文本可以稍微带一点颜色区分，但不破坏段落连贯性 */
.claim-sentence.is-insight .claim-text {
  color: #2a5caa;
  /* font-style: italic; 如果你喜欢斜体可以解开注释 */
}

/* ================= 句末角标样式 (Superscript) ================= */

.citation-badge, .insight-badge {
  cursor: pointer;
  font-size: 0.75rem;
  font-weight: 600;
  margin: 0 2px 0 4px; /* 紧贴句子末尾 */
  padding: 1px 4px;
  border-radius: 4px;
  user-select: none;
  transition: all 0.2s ease;
  vertical-align: super; /* 确保它像论文引用一样飘在右上角 */
}

/* 客观证据角标 */
.citation-badge {
  color: #409EFF;
  background-color: #ecf5ff;
  border: 1px solid #b3d8ff;
}

.citation-badge:hover {
  background-color: #409EFF;
  color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 2px 4px rgba(64, 158, 255, 0.3);
}

/* AI 洞察角标 */
.insight-badge {
  color: #e6a23c;
  background-color: #fdf6ec;
  border: 1px solid #f3d19e;
  cursor: default; /* 这个不需要点击 */
}

.sentence-spacer {
  display: inline-block;
  width: 4px;
}
</style>