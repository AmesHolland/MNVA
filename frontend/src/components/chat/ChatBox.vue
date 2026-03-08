<script setup>
import { useChatStore } from '../../store/chatStore'
import { ref, watch, nextTick } from 'vue'
import ProcessLog from './ProcessLog.vue'
import EmbeddedPlanCard from './EmbeddedPlanCard.vue'

const store = useChatStore()
const chatContainer = ref(null)

// 监听消息变化，自动滚动到底部
watch(() => store.messages.length, async () => {
  await nextTick()
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}, { deep: true })
</script>

<template>
  <div class="chat-box" ref="chatContainer">
    <div
      v-for="msg in store.messages"
      :key="msg.id"
      :class="['message-wrapper', msg.role === 'user' ? 'is-user' : 'is-ai']"
    >
      <div class="avatar" :class="msg.role === 'user' ? 'user-avatar' : 'ai-avatar'">
        {{ msg.role === 'user' ? 'User' : 'AI' }}
      </div>

      <div class="message-content-container">

        <!-- 1. 普通文本消息 -->
        <div v-if="msg.type === 'text' || !msg.type" class="message-bubble">
          {{ msg.content }}
        </div>

        <!-- 2. 思维链折叠面板 -->
        <ProcessLog
          v-else-if="msg.type === 'process'"
          :steps="msg.steps"
          :isExpanded="msg.isExpanded"
        />

        <!-- 3. 嵌入式审批卡片 -->
        <EmbeddedPlanCard
          v-else-if="msg.type === 'plan_card'"
          :messageId="msg.id"
          :planData="msg.planData"
          :isApproved="msg.isApproved"
        />

      </div>
    </div>

    <!-- 正在输入的打字机效果 -->
    <div v-if="store.isGenerating" class="message-wrapper is-ai">
       <!-- 只有当最后一条消息不是 process 类型时，才显示独立的打字机气泡 -->
       <!-- 如果最后一条是 process，说明正在更新步骤，不需要额外的打字机 -->
       <template v-if="store.messages.length > 0 && store.messages[store.messages.length-1].type !== 'process'">
          <div class="avatar ai-avatar">AI</div>
          <div class="message-bubble typing-indicator">
            <span></span><span></span><span></span>
          </div>
       </template>
    </div>
  </div>
</template>

<style scoped>
.chat-box {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-bottom: 20px;
  height: 100%;
  overflow-y: auto;
  padding-right: 10px; /* 防止滚动条遮挡内容 */
}

.message-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  text-align: left;
  width: 100%;
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
  font-size: 12px;
  font-weight: bold;
  color: white;
  flex-shrink: 0;
}

.ai-avatar { background-color: #409EFF; }
.user-avatar { background-color: #67C23A; }

.message-content-container {
  max-width: 85%;
  display: flex;
  flex-direction: column;
  /* 确保子元素（如卡片）能撑满容器宽度 */
  width: auto;
}

.message-bubble {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.6;
  word-wrap: break-word;
  white-space: pre-wrap;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.is-ai .message-bubble {
  background-color: #f4f4f5;
  color: #303133;
  border-top-left-radius: 2px;
}

.is-user .message-bubble {
  background-color: #e1f3d8;
  color: #1a4a04;
  border-top-right-radius: 2px;
}

/* 简单的打字机动画 */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 16px 20px;
}

.typing-indicator span {
  display: inline-block;
  width: 6px;
  height: 6px;
  background-color: #909399;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
</style>
