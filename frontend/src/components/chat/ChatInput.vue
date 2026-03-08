<script setup>
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()

const handleSend = () => {
  if (store.inputText.trim() && !store.isGenerating) {

    // 1. 调用更新后的 sendMessage，把暂存的沙盒指令（如果有的话）传进去
    store.sendMessage(store.inputText, store.pendingSandboxContext)

    // 2. 发送完毕后，清理战场
    store.inputText = ''
    store.pendingSandboxContext.value = {
      is_sandbox_request: false,
      sandbox_constraints: { }

    } // 消费完毕，清空暂存

    // （可选）同时清除地图和时间轴上的刷选状态，让 UI 恢复初始形态
    if (store.clearBrushState) {
        store.clearBrushState()
    }
  }
}

// 支持回车发送，Shift+Enter 换行 (保持你原有的优秀逻辑)
const handleKeyDown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="chat-input-container">
    <textarea
      v-model="store.inputText"
      @keydown="handleKeyDown"
      class="custom-textarea"
      placeholder="Enter instructions (Enter to send, Shift+Enter to wrap)..."
      :disabled="store.isGenerating"
      rows="3"
    ></textarea>

    <button
      class="send-btn"
      @click="handleSend"
      :disabled="!store.inputText.trim() || store.isGenerating"
    >
      Send
    </button>
  </div>
</template>

<style scoped>
.chat-input-container {
  display: flex;
  flex-direction: column;
  gap: 10px;
  height: 100%;
}

.custom-textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  resize: none;
  font-family: inherit;
  font-size: 14px;
  transition: border-color 0.2s;
}
.custom-textarea:focus {
  outline: none;
  border-color: #409EFF;
}
.custom-textarea:disabled {
  background-color: #f5f7fa;
  cursor: not-allowed;
}

.send-btn {
  align-self: flex-end;
  padding: 8px 24px;
  background-color: #409EFF;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: opacity 0.2s;
}
.send-btn:hover:not(:disabled) {
  opacity: 0.8;
}
.send-btn:disabled {
  background-color: #a0cfff;
  cursor: not-allowed;
}
</style>