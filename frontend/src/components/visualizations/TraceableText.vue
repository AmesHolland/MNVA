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

// 【核心魔术】：计算这句 Claim 是否应该变暗 (Dimmed)
const isClaimDimmed = (claim) => {
  const timeRange = store.brushState.timeRange
  const activeLabels = store.brushState.spatialLabels || []

  // 如果当前没有任何刷选状态，所有文本保持明亮
  if (!timeRange && activeLabels.length === 0) return false

  // 如果这句话没有证据支撑，在严格模式下我们让它变暗，以凸显有证据的文本
  if (!claim.source_ids || claim.source_ids.length === 0) return true

  const pool = store.analysisResults?.evidence_pool || {}
  const [brushStart, brushEnd] = timeRange || [0, Infinity]
  // console.log(pool[0])
  // console.log("timeRange" + timeRange)
  // 检查支撑这句话的所有底层新闻，看看有没有任何一篇命中了当前的时空沙盒
  const hasMatch = claim.source_ids.some(docId => {

    // 1. 防御性检查：跳过空的DOC_ID
    if (!docId) return false;

    // 2. 根据DOC_ID从pool数组中查找对应的新闻文档
    // 使用可选链?.防止pool中的元素为null/undefined时报错
    const doc = pool.find(item => item?.DOC_ID === docId);
    // const doc = pool[id]
    if (!doc) return false

    // 1. 时间校验
    let timeMatch = true
    if (doc.publish_date) {
      const docTime = new Date(doc.publish_date).getTime()
      timeMatch = docTime >= brushStart && docTime <= brushEnd
    }

    // 2. 空间与语义标签校验
    let labelMatch = true
    // if (activeLabels.length > 0) {
    //   // 将新闻的各个字段拼接成大字符串进行模糊匹配
    //   const textToSearch = `${doc.title} ${doc.content} ${doc.country || ''} ${(doc.locations || []).join(' ')}`.toLowerCase()
    //   // 只要有一条标签命中即可
    //   labelMatch = activeLabels.some(label => textToSearch.includes(label.toLowerCase()))
    // }

    return timeMatch && labelMatch
  })

  // 如果没有底层新闻匹配，这句话就退居背景
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
        'direct-quote': claim.is_direct_quote,
        'is-dimmed': isClaimDimmed(claim) /* 动态绑定变暗样式 */
      }"
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

/* 句子原样式与过渡动画 */
.claim-sentence {
  display: inline;
  transition: opacity 0.4s ease, color 0.4s ease, background-color 0.4s ease;
}

/* 如果是直接截取原话，给一点极其微妙的背景色提示 */
.claim-sentence.direct-quote {
  background-color: rgba(64, 158, 255, 0.08);
  border-radius: 2px;
}

/* 【核心样式】：褪色退居背景的样式 */
.claim-sentence.is-dimmed {
  opacity: 0.25;
  color: #909399;
}

/* 当句子变暗时，引用角标也顺带变暗隐藏，降低视觉干扰 */
.claim-sentence.is-dimmed .citation-badge {
  background-color: transparent;
  border-color: transparent;
  color: #c0c4cc;
  opacity: 0.5;
}

/* 角标原样式 */
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