<script setup>
import { computed } from 'vue'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()

// 响应式获取 Store 中的洞察数据
const insight = computed(() => {
  return store.analysisResults?.insight || null
})

// 判断是否为有效数据（排除我们初始化的占位符）
const hasValidData = computed(() => {
  return insight.value && insight.value.title && insight.value.title !== "Waiting data..."
})
</script>

<template>
  <div class="insight-panel-container">
    <div v-if="hasValidData" class="insight-content fade-in">
      <div class="insight-header-row">
        <h3 class="title">
          <span class="icon">💡</span> {{ insight.title }}
        </h3>
        <div class="keywords">
          <span v-for="(kw, index) in insight.keywords" :key="index" class="keyword-tag">
            # {{ kw }}
          </span>
        </div>
      </div>

      <p class="summary">{{ insight.summary }}</p>
    </div>

    <div v-else class="empty-state">
      <div v-if="store.isGenerating" class="generating-box">
        <div class="skeleton-title pulse"></div>
        <div class="skeleton-line pulse"></div>
        <div class="skeleton-line short pulse"></div>
      </div>
      <div v-else class="hint-box">
        <span class="hint-icon">🎨</span>
        <p class="hint-text">Please input your exploration objective in the left console, and the system will conduct macro situational analysis here...</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.insight-panel-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

/* --- 真实内容样式 --- */
.insight-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.title {
  margin: 0;
  font-size: 1.1rem;
  color: #303133;
  display: flex;
  align-items: center;
  gap: 6px;
}

.icon {
  font-size: 1.2rem;
}

.keywords {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-tag {
  background-color: #ecf5ff;
  color: #409eff;
  font-size: 0.75rem;
  padding: 4px 10px;
  border-radius: 14px;
  font-weight: 500;
  border: 1px solid #d9ecff;
}

.summary {
  margin: 0;
  font-size: 0.9rem;
  color: #606266;
  line-height: 1.6;
  text-align: justify;
}

/* --- 占位与骨架屏样式 --- */
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.hint-box {
  display: flex;
  align-items: center;
  gap: 10px;
}

.hint-icon {
  font-size: 1.5rem;
  opacity: 0.5;
}

.hint-text {
  font-size: 0.9rem;
  margin: 0;
}

.generating-box {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.skeleton-title {
  height: 20px;
  width: 40%;
  background-color: #e4e7ed;
  border-radius: 4px;
  margin-bottom: 8px;
}

.skeleton-line {
  height: 14px;
  width: 100%;
  background-color: #ebeef5;
  border-radius: 4px;
}

.skeleton-line.short {
  width: 70%;
}

/* 动画效果 */
.pulse {
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
}

.fade-in {
  animation: fadeIn 0.4s ease-out forwards;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>