<script setup>
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()

const handleSend = () => {
  if (store.inputText.trim() && !store.isGenerating) {

    // 1. 发送消息，附带隐式的沙盒上下文
    store.sendMessage(store.inputText, store.pendingSandboxContext)

    // 2. 发送完毕后，清理战场
    store.inputText = ''

    // 【修正】：不要使用 .value 去重置 Pinia state 中的对象，直接覆盖属性或整个对象
    store.pendingSandboxContext = {
      is_sandbox_request: false,
      sandbox_constraints: {}
    }

    // 3. 清除全局刷选状态，让 UI 恢复初始形态
    if (typeof store.clearBrushState === 'function') {
        store.clearBrushState()
    }
  }
}

const handleKeyDown = (e) => {
  // 保持回车发送，Shift+Enter 换行
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="chat-input-container">
    <transition name="fade">
      <div v-if="store.pendingSandboxContext?.is_sandbox_request" class="sandbox-indicator">
        <span class="sandbox-icon">📍</span>
        <span class="sandbox-text">Local Spatiotemporal Sandbox Active</span>
        <button class="cancel-sandbox-btn" @click="store.pendingSandboxContext.is_sandbox_request = false" title="Cancel constraints">×</button>
      </div>
    </transition>

    <div class="input-wrapper">
      <textarea
        v-model="store.inputText"
        @keydown="handleKeyDown"
        :class="['custom-textarea', { 'has-sandbox': store.pendingSandboxContext?.is_sandbox_request }]"
        placeholder="Enter analysis instructions (Enter to send, Shift+Enter to wrap)..."
        :disabled="store.isGenerating"
        rows="3"
      ></textarea>

      <button
        class="send-btn"
        @click="handleSend"
        :disabled="!store.inputText.trim() || store.isGenerating"
        title="Send Request"
      >
        <svg viewBox="0 0 24 24" class="send-icon" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z"/></svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-input-container {
  display: flex;
  flex-direction: column;
  gap: 8px; /* 紧凑布局 */
  width: 100%;
}

/* --- 沙盒模式指示器 --- */
.sandbox-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background-color: #f0f9ff; /* 极其克制的浅蓝底色 */
  border: 1px solid #bae6fd;
  border-radius: 6px;
  font-size: 0.8rem;
  color: #0369a1;
}

.sandbox-icon { font-size: 0.9rem; }
.sandbox-text { flex: 1; font-weight: 500; }

.cancel-sandbox-btn {
  background: transparent;
  border: none;
  color: #0284c7;
  cursor: pointer;
  font-size: 1.2rem;
  line-height: 1;
  padding: 0 4px;
  transition: color 0.2s;
}
.cancel-sandbox-btn:hover { color: #dc2626; } /* 悬浮时变红警告 */

/* --- 输入框区域 --- */
.input-wrapper {
  position: relative;
  width: 100%;
}

.custom-textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 12px 45px 12px 12px; /* 右侧留出按钮空间 */
  border: 1px solid #e2e8f0; /* 学术冷灰边框 */
  border-radius: 6px;
  resize: none;
  font-family: inherit;
  font-size: 0.95rem;
  color: #334155;
  background-color: #ffffff;
  transition: all 0.2s ease;
  line-height: 1.5;
}

/* 普通聚焦 */
.custom-textarea:focus {
  outline: none;
  border-color: #94a3b8;
  box-shadow: 0 0 0 1px #94a3b8;
}

/* 沙盒模式下的聚焦：边框变为主题蓝，强化感知 */
.custom-textarea.has-sandbox {
  border-color: #7dd3fc;
}
.custom-textarea.has-sandbox:focus {
  border-color: #0ea5e9;
  box-shadow: 0 0 0 1px #0ea5e9;
}

.custom-textarea:disabled {
  background-color: #f8fafc;
  color: #94a3b8;
  cursor: not-allowed;
}

/* 现代化的悬浮发送按钮 */
.send-btn {
  position: absolute;
  right: 8px;
  bottom: 8px;
  width: 32px;
  height: 32px;
  padding: 6px;
  background-color: #2563eb; /* 专业主操作蓝 */
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.send-icon {
  width: 18px;
  height: 18px;
  transform: translateY(1px) translateX(-1px); /* 微调视觉中心 */
}

.send-btn:hover:not(:disabled) {
  background-color: #1d4ed8;
}

.send-btn:disabled {
  background-color: #cbd5e1;
  cursor: not-allowed;
}

/* Vue 动画 */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
  transform: translateY(5px);
}
</style>