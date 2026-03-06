<script setup>
import { computed , ref} from 'vue'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()
// 展开状态：key = DOC_ID, value = 是否展开全部内容
const expandedStates = ref({})
// 计算属性：1. 加载全部证据 2. ids中的项高亮+排到最前面 3. 其余项按ID排序
const allEvidences = computed(() => {
  // 1. 获取全部证据池数据（数组）
  const pool = store.analysisResults?.evidence_pool || []
  // 2. 获取需要高亮的ID列表
  const activeIds = store.evidenceState.activeSourceIds || []

  // 3. 拆分数据：活跃项（在activeIds中）、非活跃项（不在activeIds中）
  const activeItems = []
  const inactiveItems = []

  pool.forEach(item => {
    // 给每个项标记是否活跃（用于高亮和排序）
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

  // 4. 排序规则：
  // - 活跃项：按 ID 排序后放前面
  // - 非活跃项：按 ID 排序后放后面
  const sortById = (a, b) => a.id.localeCompare(b.id, undefined, { numeric: true })
  activeItems.sort(sortById)
  inactiveItems.sort(sortById)

  // 5. 合并：活跃项在前，非活跃项在后
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

      <div class="drawer-body">
        <!-- 显示总数 + 活跃数 -->
        <p class="drawer-hint">
          Total {{ allEvidences.length }} intelligence records (including {{ store.evidenceState.activeSourceIds?.length || 0 }} core supporting items):
        </p>

        <div class="evidence-list">
          <!-- 遍历全部证据，根据isActive判断是否高亮 -->
          <div
            v-for="item in allEvidences"
            :key="item.id"
            class="evidence-card"
            :class="{ 'is-highlight': item.isActive }"
          >
            <div class="card-meta">
              <span class="meta-tag source-tag">{{ item.source || 'Unknown Source' }}</span>
              <span class="meta-tag date-tag">{{ item.publish_date || 'N/A' }}</span>
              <!-- 新增：标记是否为核心支撑项 -->
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
              <!-- 内容过长时显示展开/收起字样 -->
              <span
                v-if="item.content && item.content.length > 150"
                class="toggle-content"
                @click="toggleExpand(item.id)"
              >
                {{ expandedStates[item.id] ? 'Collapse' : 'Expand all' }}
              </span>
            </div>
            <div class="doc-id">DOC ID: {{ item.id }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 原有样式保留，新增/修改以下部分 */
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
  transition: all 0.2s ease; /* 高亮过渡动画 */
  text-align: left;
}
/* 新增：高亮样式 */
.evidence-card.is-highlight {
  border-color: #409EFF;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
  background: linear-gradient(180deg, #f0f7ff 0%, #ffffff 100%);
}
/* 高亮卡片的内容区样式强化 */
.evidence-card.is-highlight .card-content {
  border-left-color: #409EFF;
  background: #ecf5ff;

}

.card-meta { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;  text-align: left; /* 新增：元标签左对齐 */}
.meta-tag { font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; }
.source-tag { background: #ecf5ff; color: #409EFF; border: 1px solid #b3d8ff; font-weight: 500;}
.date-tag { background: #f4f4f5; color: #909399; border: 1px solid #e9e9eb; }
/* 新增：核心支撑标签样式 */
.active-tag { background: #fdf2e8; color: #F56C6C; border: 1px solid #fbc4ab; }

.card-title { margin: 0 0 10px 0; font-size: 1rem; color: #303133; line-height: 1.4; }
.geo-tags { font-size: 0.8rem; color: #E6A23C; margin-bottom: 10px; font-weight: 500;}
.card-content {
  font-size: 0.9rem; color: #606266; line-height: 1.6;
  background: #fafafa; padding: 10px; border-radius: 4px; border-left: 3px solid #dcdfe6;
  transition: all 0.2s ease;
}
/* 新增：展开/收起字样样式 */
.toggle-content {
  color: gray;
  cursor: pointer;
  font-weight: 500;
  text-decoration: underline;
  margin-left: 4px;
}
.toggle-content:hover {
  color: #66b1ff;
}
.doc-id { text-align: right; font-size: 0.7rem; color: #c0c4cc; margin-top: 10px; font-family: monospace; }
</style>