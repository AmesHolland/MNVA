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
/* 容器布局保持清爽 */
.chat-box {
  display: flex;
  flex-direction: column;
  gap: 20px; /* 稍微增加间距，让对话呼吸感更强 */
  padding-bottom: 24px;
  height: 100%;
  overflow-y: auto;
  padding-right: 12px;
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

/* 头像设计：摒弃高饱和色块，采用学术风冷色调 */
.avatar {
  width: 32px; /* 稍微缩小头像，突出内容 */
  height: 32px;
  border-radius: 6px; /* 从纯圆改为微圆角矩形，更具工具感 */
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  flex-shrink: 0;
  border: 1px solid transparent;
}

/* AI 探员的头像：专业冷灰 */
.ai-avatar {
  background-color: #f1f5f9;
  color: #475569;
  border-color: #e2e8f0;
}

/* 用户的头像：克制的主题浅蓝 */
.user-avatar {
  background-color: #e0f2fe;
  color: #0369a1;
  border-color: #bae6fd;
}

.message-content-container {
  max-width: 88%;
  display: flex;
  flex-direction: column;
  width: auto;
}

/* 气泡基础样式：取消阴影，使用极细边框界定边界 */
.message-bubble {
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 0.95rem; /* 稍微精细化字号 */
  line-height: 1.6;
  word-wrap: break-word;
  white-space: pre-wrap;
  color: #334155; /* 统一使用深石板灰作为主要阅读色 */
}

/* AI 回复气泡：极简灰白底色，像日志卡片 */
.is-ai .message-bubble {
  background-color: #ffffff;
  border: 1px solid #e2e8f0;
  border-top-left-radius: 2px; /* 保留一点方向性暗示 */
}

/* User 指令气泡：极其微弱的浅蓝底色，表明主动输入属性 */
.is-user .message-bubble {
  background-color: #f0f9ff;
  border: 1px solid #e0f2fe;
  border-top-right-radius: 2px;
}

/* 打字机动画：弱化视觉抢夺 */
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 12px 16px;
  background-color: transparent !important;
  border: none !important;
}

.typing-indicator span {
  display: inline-block;
  width: 5px;
  height: 5px;
  background-color: #cbd5e1; /* 极浅的灰色 */
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out both;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}
</style>
