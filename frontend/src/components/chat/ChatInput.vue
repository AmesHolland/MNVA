<script setup>
import { ref } from 'vue'
import { useChatStore } from '../../store/chatStore'

const store = useChatStore()
const inputText = ref('')

const handleSend = () => {
  if (inputText.value.trim() && !store.isGenerating && !store.hitlState.isWaiting) {
    store.sendMessage(inputText.value)
    inputText.value = ''
  }
}

// 支持回车发送，Shift+Enter 换行
const handleKeyDown = (e) => {
  // 确保是在没有使用输入法组合键（isComposing）的情况下按下的回车
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="chat-input-container">
    <textarea
      v-model="inputText"
      @keydown="handleKeyDown"
      class="custom-textarea"
      placeholder="Enter instructions (Enter to send, Shift+Enter to wrap)..."
      :disabled="store.isGenerating || store.hitlState.isWaiting"
      rows="3"
    ></textarea>

    <button
      class="send-btn"
      @click="handleSend"
      :disabled="!inputText.trim() || store.isGenerating || store.hitlState.isWaiting"
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