<script setup>
import { ref } from 'vue'
import { useChatStore } from '../../store/chatStore'

const props = defineProps({
  messageId: {
    type: Number,
    required: true
  },
  planData: {
    type: Object,
    required: true
  },
  isApproved: {
    type: Boolean,
    default: false
  }
})

const store = useChatStore()
const feedbackText = ref('')
const isSubmitting = ref(false)
const editMode = ref(false) // 引入 ApprovalCard 的交互模式

const handleApprove = async () => {
  if (props.isApproved || isSubmitting.value) return
  isSubmitting.value = true
  await store.sendFeedback('approve', props.messageId)
  isSubmitting.value = false
}

const handleReject = async () => {
  if (props.isApproved || isSubmitting.value) return

  if (!feedbackText.value.trim()) {
    alert('Please enter revision feedback')
    return
  }

  isSubmitting.value = true
  await store.sendFeedback(feedbackText.value, props.messageId)
  isSubmitting.value = false
  editMode.value = false // 提交后重置状态
}
</script>

<template>
  <div class="approval-card" :class="{ 'disabled': isApproved }">
    <div class="card-header">
      <div class="header-title">
        <span class="icon"></span> Plan
      </div>
      <span class="status-badge" v-if="isApproved">✅ Approved</span>
    </div>

    <div class="card-body">
      <div class="plan-summary">
        <strong>Plan Logic:</strong>
        <p>{{ planData.total_plan_logic }}</p>
      </div>

      <div class="task-list">
        <div
          v-for="(task, index) in planData.tasks"
          :key="task.task_id"
          class="task-item"
        >
          <div class="task-status">
            <div class="circle-check">✓</div>
            <div v-if="index !== planData.tasks.length - 1" class="connecting-line"></div>
          </div>

          <div class="task-content">
            <div class="task-agent">
              <span class="agent-badge">{{ task.agent.replace('_Agent', '') }}</span>

            </div>
            <div class="task-action">{{ task.action }}</div>

            <div class="task-args" v-if="task.args && (task.args.keywords || task.args.target_entity || task.args.time_range)">
              <span class="arg-tag" v-if="task.args.keywords">
                🗝️ {{ task.args.keywords }}
              </span>
              <span class="arg-tag" v-if="task.args.target_entity">
                🎯 {{ task.args.target_entity }}
              </span>
              <span class="arg-tag" v-if="task.args.time_range">
                ⏳ {{ task.args.time_range }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card-actions" v-if="!isApproved">
      <template v-if="!editMode">
        <button class="btn btn-primary" @click="handleApprove" :disabled="isSubmitting">
          {{ isSubmitting ? 'Processing...' : 'Run' }}
        </button>
        <button class="btn btn-text" @click="editMode = true" :disabled="isSubmitting">Deny</button>
      </template>

      <template v-else>
        <textarea
          v-model="feedbackText"
          placeholder="Please enter intervention instructions, e.g.: 'Remove the analysis of NOAA and add tracking of the U.S. Navy'..."
          class="feedback-input"
          rows="3"
          :disabled="isSubmitting"
        ></textarea>
        <div class="edit-actions">
            <button class="btn btn-primary" @click="handleReject" :disabled="isSubmitting">
              {{ isSubmitting ? 'Submitting...' : 'Submit Replan' }}
            </button>
            <button class="btn btn-text" @click="editMode = false" :disabled="isSubmitting">Cancel</button>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
/* 融合两者的容器样式 */
.approval-card {
  background-color: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  margin: 12px 0;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  text-align: left;
  transition: all 0.3s;
  max-width: 100%;
}

.approval-card.disabled {
  opacity: 0.8;
  pointer-events: none;
  background-color: #f5f7fa;
}

.card-header {
  background-color: #fafafa;
  color: #303133;
  padding: 12px 15px;
  font-weight: 600;
  font-size: 14px;
  border-bottom: 1px solid #ebeef5;
  display: flex;
  align-items: center;
  justify-content: space-between; /* 支持两侧对齐 */
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-badge {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 12px;
  background-color: #f0f9eb;
  color: #67C23A;
  border: 1px solid #e1f3d8;
  font-weight: 500;
}

.card-body {
  padding: 15px;
}

/* 规划总览样式 */
.plan-summary {
  background-color: #f3f4f6;
  padding: 10px;
  border-radius: 6px;
  margin-bottom: 15px;
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}
.plan-summary strong {
  color: #303133;
  display: block;
  margin-bottom: 4px;
}
.plan-summary p {
  margin: 0;
}

/* 任务列表样式 */
.task-list {
  display: flex;
  flex-direction: column;
}

.task-item {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
}

.task-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 20px;
}

.circle-check {
  width: 18px;
  height: 18px;
  background-color: #333;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  z-index: 2;
}

.connecting-line {
  width: 2px;
  flex-grow: 1;
  background-color: #e4e7ed;
  margin-top: 4px;
  min-height: 20px;
}

.task-content {
  background-color: #f9fafc;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  padding: 10px;
  flex-grow: 1;
}

.task-agent {
  margin-bottom: 6px;
}

.agent-badge {
  background-color: #409EFF;
  color: white;
  font-size: 12px;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}

.dependency-text {
  font-size: 12px;
  color: #909399;
  margin-left: 8px;
}

.task-action {
  font-size: 13px;
  color: #303133;
  line-height: 1.4;
  margin-bottom: 8px;
}

.task-args {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.arg-tag {
  background-color: #ecf5ff;
  color: #409eff;
  border: 1px solid #b3d8ff;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 4px;
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

}

/* 底部按钮区 */
.card-actions {
  padding: 12px 15px;
  border-top: 1px solid #ebeef5;
  background-color: #fafafa;
  display: flex;
  gap: 10px;
  justify-content: center;
}

.btn {
  padding: 6px 20px;
  border: none;
  border-radius: 16px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.btn-primary:not(:disabled):hover { background-color: #dcfce7; }
.btn-text { background-color: #ffffff; color: #374151; border: 1px solid #e5e7eb; }
.btn-text:not(:disabled):hover { background-color: #f3f4f6; }

.feedback-input {
  width: 100%;
  box-sizing: border-box;
  padding: 8px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-family: inherit;
  margin-bottom: 10px;
  font-size: 13px;
  resize: vertical;
}

.feedback-input:focus {
  border-color: #409EFF;
  outline: none;
}

.edit-actions { display: flex; gap: 10px; justify-content: flex-end; width: 100%; }
</style>