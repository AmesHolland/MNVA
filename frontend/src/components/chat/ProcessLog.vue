<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  steps: {
    type: Array,
    default: () => []
  },
  isExpanded: {
    type: Boolean,
    default: true
  }
})

const isOpen = ref(props.isExpanded)

const toggle = () => {
  isOpen.value = !isOpen.value
}

// 计算总耗时（仅作展示用，实际需后端传时间戳）
const duration = computed(() => {
  if (props.steps.length < 2) return 'Calculating...'
  const start = props.steps[0].timestamp
  const end = props.steps[props.steps.length - 1].timestamp
  return ((end - start) / 1000).toFixed(1) + 's'
})
</script>

<template>
  <div class="process-log-container">
    <div class="log-header" @click="toggle">
      <span class="status-icon" :class="{ 'spinning': props.steps.length > 0 && props.steps[props.steps.length-1].status === 'loading' }">
        ⚙️
      </span>
      <span class="title">Thinking Process</span>
      <span class="duration" v-if="!isOpen">{{ duration }}</span>
      <span class="arrow" :class="{ 'open': isOpen }">▼</span>
    </div>

    <transition name="expand">
      <div v-if="isOpen" class="log-body">
        <div v-for="(step, index) in steps" :key="index" class="log-step">
          <div class="step-line"></div>
          <div class="step-dot" :class="{ 'active': index === steps.length - 1 }"></div>
          <div class="step-content">
            <span class="step-name">{{ step.node }}</span>
            <span class="step-status" v-if="index < steps.length - 1">✓</span>
            <span class="step-status loading" v-else>...</span>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.process-log-container {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  background-color: #f9fafc;
  margin: 8px 0;
  overflow: hidden;
  font-family: 'Consolas', monospace;
  font-size: 0.85rem;
}

.log-header {
  padding: 8px 12px;
  background-color: #f4f4f5;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
  transition: background-color 0.2s;
}

.log-header:hover {
  background-color: #e9e9eb;
}

.title {
  font-weight: 600;
  color: #606266;
  flex: 1;
  margin-left: 8px;
}

.duration {
  color: #909399;
  font-size: 0.75rem;
  margin-right: 8px;
}

.arrow {
  font-size: 0.7rem;
  color: #909399;
  transition: transform 0.3s;
}

.arrow.open {
  transform: rotate(180deg);
}

.log-body {
  padding: 12px 16px;
  background-color: #ffffff;
  border-top: 1px solid #e4e7ed;
}

.log-step {
  display: flex;
  align-items: center;
  position: relative;
  padding-bottom: 12px;
}

.log-step:last-child {
  padding-bottom: 0;
}

.step-line {
  position: absolute;
  left: 5px;
  top: 14px;
  bottom: -4px;
  width: 2px;
  background-color: #e4e7ed;
  z-index: 0;
}

.log-step:last-child .step-line {
  display: none;
}

.step-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #dcdfe6;
  z-index: 1;
  margin-right: 12px;
  border: 2px solid #ffffff;
  box-shadow: 0 0 0 1px #dcdfe6;
}

.step-dot.active {
  background-color: #409EFF;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
  animation: pulse 1.5s infinite;
}

.step-content {
  flex: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.step-name {
  color: #303133;
}

.step-status {
  color: #67C23A;
  font-weight: bold;
}

.step-status.loading {
  color: #409EFF;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(64, 158, 255, 0.4); }
  70% { box-shadow: 0 0 0 6px rgba(64, 158, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(64, 158, 255, 0); }
}

.spinning {
  animation: spin 2s linear infinite;
  display: inline-block;
}

@keyframes spin { 100% { transform: rotate(360deg); } }

/* 折叠动画 */
.expand-enter-active, .expand-leave-active {
  transition: all 0.3s ease;
  max-height: 500px;
  opacity: 1;
}
.expand-enter-from, .expand-leave-to {
  max-height: 0;
  opacity: 0;
  padding-top: 0;
  padding-bottom: 0;
}
</style>
