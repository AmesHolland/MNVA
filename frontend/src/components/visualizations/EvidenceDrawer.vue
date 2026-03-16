<script setup>
import { computed, ref } from 'vue'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()

// 展开状态：key = DOC_ID, value = 是否展开全部内容
const expandedStates = ref({})

// 计算属性：1. 加载全部证据 2. ids中的项高亮+排到最前面 3. 其余项按ID排序
const allEvidences = computed(() => {
  const pool = store.analysisResults?.evidence_pool || []
  const activeIds = store.evidenceState.activeSourceIds || []

  const activeItems = []
  const inactiveItems = []

  pool.forEach(item => {
    const isActive = activeIds.includes(item.DOC_ID)
    const evidenceItem = {
      id: item.DOC_ID, // 统一id字段
      isActive,       // 标记是否高亮
      ...item
    }

    if (isActive) {
      activeItems.push(evidenceItem)
    } else {
      inactiveItems.push(evidenceItem)
    }
  })

  const sortById = (a, b) => a.id.localeCompare(b.id, undefined, { numeric: true })
  activeItems.sort(sortById)
  inactiveItems.sort(sortById)

  return [...activeItems, ...inactiveItems]
})

// 切换展开/收起状态
const toggleExpand = (id) => {
  expandedStates.value[id] = !expandedStates.value[id]
}

// 截断内容的辅助函数
const truncateContent = (content, maxLength = 150) => {
  if (!content || content.length <= maxLength) return content
  return content.slice(0, maxLength) + '...'
}

// 🌟 新增 1：处理用户的打标点击
const handleAnnotate = (sourceId, type) => {
  const claimId = store.evidenceState.activeClaimId
  if (!claimId) return
  store.toggleEvidenceAnnotation(claimId, sourceId, type)
}

// 🌟 新增 2：计算按钮的激活状态样式
const getBtnClass = (sourceId, type) => {
  const claimId = store.evidenceState.activeClaimId
  if (!claimId) return ''
  const status = store.getAnnotationStatus(claimId, sourceId)
  return status === type ? `is-active active-${type}` : ''
}
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
        <h3>Evidence Sources</h3>
        <button class="close-btn" @click="store.closeEvidence">×</button>
      </header>

      <div class="context-alert" v-if="store.evidenceState.activeClaimText">
        <span class="context-label">Current Claim Context:</span>
        <p class="context-text">"{{ store.evidenceState.activeClaimText }}"</p>
      </div>

      <div class="drawer-body">
        <p class="drawer-hint">
          Total {{ allEvidences.length }} intelligence records (including {{ store.evidenceState.activeSourceIds?.length || 0 }} core supporting items):
        </p>

        <div class="evidence-list">
          <div
            v-for="item in allEvidences"
            :key="item.id"
            class="evidence-card"
            :class="{ 'is-highlight': item.isActive }"
          >
            <div class="card-meta">
              <span class="meta-tag source-tag">{{ item.source || 'Unknown Source' }}</span>
              <span class="meta-tag date-tag">{{ item.publish_date || 'N/A' }}</span>
              <span v-if="item.isActive" class="meta-tag active-tag">Core Support</span>
            </div>

            <h4 class="card-title">{{ item.title || 'No title' }}</h4>

            <div class="geo-tags" v-if="item.country || item.region">
              📍 {{ item.country }} | {{ item.region }}
            </div>

            <div class="card-content">
              {{
                expandedStates[item.id]
                  ? item.content
                  : truncateContent(item.content)
              }}
              <span
                v-if="item.content && item.content.length > 150"
                class="toggle-content"
                @click="toggleExpand(item.id)"
              >
                {{ expandedStates[item.id] ? 'Collapse' : 'Expand all' }}
              </span>
            </div>

            <div class="doc-id">DOC ID: {{ item.id }}</div>

            <footer class="annotation-actions" v-if="item.isActive && store.evidenceState.activeClaimId">
              <span class="action-hint">Is this evidence properly used?</span>
              <div class="btn-group">
                <button
                  class="annotate-btn btn-irrelevant"
                  :class="getBtnClass(item.id, 'irrelevant')"
                  @click="handleAnnotate(item.id, 'irrelevant')"
                  title="Mark as irrelevant to the current claim"
                >
                   Irrelevant
                </button>
                <button
                  class="annotate-btn btn-contradicts"
                  :class="getBtnClass(item.id, 'contradicts')"
                  @click="handleAnnotate(item.id, 'contradicts')"
                  title="Mark as contradicting the current claim"
                >
                   Contradicts
                </button>
              </div>
            </footer>

          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ================= 原有样式保留 ================= */
.drawer-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(0, 0, 0, 0.4); z-index: 998;
  opacity: 0; pointer-events: none; transition: opacity 0.3s ease;
}
.drawer-overlay.is-open { opacity: 1; pointer-events: auto; }

.evidence-drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: 450px;
  background-color: #f5f7fa; box-shadow: -4px 0 16px rgba(0, 0, 0, 0.1);
  z-index: 999; transform: translateX(100%);
  transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  display: flex; flex-direction: column;
}
.evidence-drawer.is-open { transform: translateX(0); }

.drawer-header {
  padding: 20px; background-color: #ffffff; border-bottom: 1px solid #e4e7ed;
  display: flex; justify-content: space-between; align-items: center;
}
.drawer-header h3 { margin: 0; font-size: 1.1rem; color: #303133; }
.close-btn { background: none; border: none; font-size: 1.8rem; color: #909399; cursor: pointer; line-height: 1; }
.close-btn:hover { color: #F56C6C; }

.drawer-body { padding: 20px; overflow-y: auto; flex: 1; }
.drawer-hint { font-size: 0.9rem; color: #909399; margin-bottom: 15px; }

.evidence-list { display: flex; flex-direction: column; gap: 15px; }
.evidence-card {
  background: #ffffff; border: 1px solid #e4e7ed; border-radius: 8px;
  padding: 15px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
  transition: all 0.2s ease; text-align: left;
}

.evidence-card.is-highlight {
  border-color: #409EFF; box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
  background: linear-gradient(180deg, #f0f7ff 0%, #ffffff 100%);
}
.evidence-card.is-highlight .card-content { border-left-color: #409EFF; background: #ecf5ff; }

.card-meta { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; text-align: left; }
.meta-tag { font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; }
.source-tag { background: #ecf5ff; color: #409EFF; border: 1px solid #b3d8ff; font-weight: 500;}
.date-tag { background: #f4f4f5; color: #909399; border: 1px solid #e9e9eb; }
.active-tag { background: #fdf2e8; color: #F56C6C; border: 1px solid #fbc4ab; }

.card-title { margin: 0 0 10px 0; font-size: 1rem; color: #303133; line-height: 1.4; }
.geo-tags { font-size: 0.8rem; color: #E6A23C; margin-bottom: 10px; font-weight: 500;}
.card-content {
  font-size: 0.9rem; color: #606266; line-height: 1.6;
  background: #fafafa; padding: 10px; border-radius: 4px; border-left: 3px solid #dcdfe6;
  transition: all 0.2s ease;
}

.toggle-content { color: gray; cursor: pointer; font-weight: 500; text-decoration: underline; margin-left: 4px; }
.toggle-content:hover { color: #66b1ff; }
.doc-id { text-align: right; font-size: 0.7rem; color: #c0c4cc; margin-top: 10px; font-family: monospace; }

/* ================= 🌟 新增样式 ================= */

/* 上下文提示区 */
.context-alert {
  background: #ecf5ff; border-left: 4px solid #409EFF; padding: 12px 20px; font-size: 0.9rem;
  border-bottom: 1px solid #d9ecff;
}
.context-label { color: #409EFF; font-weight: bold; font-size: 0.8rem; display: block; margin-bottom: 4px; }
.context-text { margin: 0; color: #303133; font-style: italic; line-height: 1.4; }

/* 证据协商操作区 */
.annotation-actions {
  border-top: 1px dashed #ebeef5; padding-top: 12px; margin-top: 12px;
  display: flex; justify-content: space-between; align-items: center;
}
.action-hint { font-size: 0.8rem; color: #909399; font-style: italic; }
.btn-group { display: flex; gap: 8px; }

.annotate-btn {
  display: flex; align-items: center; gap: 4px; padding: 4px 10px;
  border-radius: 16px; font-size: 0.8rem; font-weight: 500;
  border: 1px solid #dcdfe6; background: #fff; color: #606266;
  cursor: pointer; transition: all 0.2s ease;
}
.annotate-btn:hover { background: #f4f4f5; }

/* 激活状态：不相关 (淡橙色) */
.annotate-btn.active-irrelevant {
  background: #fdf6ec; border-color: #f3d19e; color: #e6a23c;
  box-shadow: 0 0 0 2px rgba(230,162,60,0.1);
}

/* 激活状态：矛盾 (淡红色) */
.annotate-btn.active-contradicts {
  background: #fef0f0; border-color: #fab6b6; color: #f56c6c;
  box-shadow: 0 0 0 2px rgba(245,108,108,0.1);
}
</style>