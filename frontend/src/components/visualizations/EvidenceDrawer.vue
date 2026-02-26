<script setup>
import { computed } from 'vue'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()

// 计算属性：根据 activeSourceIds 从 evidence_pool 中提取真实的新闻数据
const activeEvidences = computed(() => {
  const pool = store.analysisResults?.evidence_pool || {}
  const ids = store.evidenceState.activeSourceIds || []

  return ids.map(id => {
    return pool[id] ? { id, ...pool[id] } : { id, title: '未找到原文记录', content: '可能已被过滤或数据异常', source: '未知' }
  })
})
</script>

<template>
  <div>
    <div
      class="drawer-overlay"
      :class="{ 'is-open': store.evidenceState.isOpen }"
      @click="store.closeEvidence"
    ></div>

    <div
      class="evidence-drawer"
      :class="{ 'is-open': store.evidenceState.isOpen }"
    >
      <header class="drawer-header">
        <h3>📄 情报溯源 (Evidence Sources)</h3>
        <button class="close-btn" @click="store.closeEvidence">×</button>
      </header>

      <div class="drawer-body">
        <p class="drawer-hint">共找到 {{ activeEvidences.length }} 条支撑该论点的情报记录：</p>

        <div class="evidence-list">
          <div v-for="item in activeEvidences" :key="item.id" class="evidence-card">
            <div class="card-meta">
              <span class="meta-tag source-tag">{{ item.source || 'Unknown Source' }}</span>
              <span class="meta-tag date-tag">{{ item.publish_date || 'N/A' }}</span>
            </div>
            <h4 class="card-title">{{ item.title }}</h4>

            <div class="geo-tags" v-if="item.country || item.region">
              📍 {{ item.country }} | {{ item.region }}
            </div>

            <div class="card-content">
              {{ item.content }}
            </div>
            <div class="doc-id">DOC ID: {{ item.id }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 遮罩层 */
.drawer-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(0, 0, 0, 0.4);
  z-index: 998;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s ease;
}
.drawer-overlay.is-open {
  opacity: 1;
  pointer-events: auto;
}

/* 抽屉面板 */
.evidence-drawer {
  position: fixed;
  top: 0; right: 0; bottom: 0;
  width: 450px;
  background-color: #f5f7fa;
  box-shadow: -4px 0 16px rgba(0, 0, 0, 0.1);
  z-index: 999;
  transform: translateX(100%); /* 默认隐藏在屏幕右侧 */
  transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  display: flex;
  flex-direction: column;
}
.evidence-drawer.is-open {
  transform: translateX(0);
}

.drawer-header {
  padding: 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.drawer-header h3 { margin: 0; font-size: 1.1rem; color: #303133; }
.close-btn { background: none; border: none; font-size: 1.8rem; color: #909399; cursor: pointer; line-height: 1; }
.close-btn:hover { color: #F56C6C; }

.drawer-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}
.drawer-hint { font-size: 0.9rem; color: #909399; margin-bottom: 15px; }

/* 原始证据卡片样式 */
.evidence-list { display: flex; flex-direction: column; gap: 15px; }
.evidence-card {
  background: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}
.card-meta { display: flex; gap: 8px; margin-bottom: 10px; }
.meta-tag { font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; }
.source-tag { background: #ecf5ff; color: #409EFF; border: 1px solid #b3d8ff; font-weight: 500;}
.date-tag { background: #f4f4f5; color: #909399; border: 1px solid #e9e9eb; }

.card-title { margin: 0 0 10px 0; font-size: 1rem; color: #303133; line-height: 1.4; }
.geo-tags { font-size: 0.8rem; color: #E6A23C; margin-bottom: 10px; font-weight: 500;}
.card-content {
  font-size: 0.9rem; color: #606266; line-height: 1.6;
  background: #fafafa; padding: 10px; border-radius: 4px; border-left: 3px solid #dcdfe6;
}
.doc-id { text-align: right; font-size: 0.7rem; color: #c0c4cc; margin-top: 10px; font-family: monospace; }
</style>