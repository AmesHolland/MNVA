<script setup>
import { useChatStore } from '../../store/chatStore'
import { ref, watch, nextTick } from 'vue'

const store = useChatStore()
const chatContainer = ref(null)

// 监听消息变化，自动滚动到底部
watch(() => store.messages.length, async () => {
  await nextTick()
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
})
</script>

<template>
  <div class="chat-box" ref="chatContainer">
    <div
      v-for="msg in store.messages"
      :key="msg.id"
      :class="['message-wrapper', msg.role === 'user' ? 'is-user' : 'is-ai']"
    >
      <div class="avatar">{{ msg.role === 'user' ? 'User' : 'AI' }}</div>

      <div class="message-content">
        {{ msg.content }}
      </div>
    </div>

    <div v-if="store.isGenerating" class="message-wrapper is-ai">
      <div class="avatar">AI</div>
      <div class="message-content typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-box {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-bottom: 20px;
}

.message-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  text-align: left;
}

.message-wrapper.is-user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  color: white;
  flex-shrink: 0;
}

.is-ai .avatar { background-color: #409EFF; }
.is-user .avatar { background-color: #67C23A; }

.message-content {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;
  white-space: pre-wrap; /* 允许换行 */
}

.is-ai .message-content {
  background-color: #f4f4f5;
  color: #303133;
  border-top-left-radius: 2px;
}

.is-user .message-content {
  background-color: #e1f3d8;
  color: #1a4a04;
  border-top-right-radius: 2px;
}

/* 简单的打字机动画 */
.typing-indicator span {
  display: inline-block;
  width: 6px;
  height: 6px;
  background-color: #909399;
  border-radius: 50%;
  margin: 0 2px;
  animation: typing 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
</style>