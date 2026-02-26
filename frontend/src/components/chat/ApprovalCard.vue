<script setup>
import { useChatStore } from '../../store/chatStore'
import { ref, watch } from 'vue'

const store = useChatStore()
const editMode = ref(false)
const localFeedback = ref('')

// 当卡片出现时，将默认反馈清空，退出编辑模式
watch(() => store.hitlState.isWaiting, (newVal) => {
  if (newVal) {
    localFeedback.value = ''
    editMode.value = false
  }
})

const handleApprove = () => {
  // 核心精简：只需调用 store 的方法，无需任何 setTimeout 或额外 push
  store.sendFeedback('approve')
}

const handleReject = () => {
  if (!localFeedback.value.trim()) {
    alert('请输入修改意见')
    return
  }
  // 发送修改意见
  store.sendFeedback(localFeedback.value)
}
</script>

<template>
  <div v-if="store.hitlState.isWaiting && store.hitlState.plan" class="approval-card">
    <div class="card-header">
      <span class="icon">🍎</span> 任务分解与规划 (Task Decomposition)
    </div>

    <div class="card-body">
      <div class="plan-summary">
        <strong>规划思路:</strong>
        <p>{{ store.hitlState.plan.total_plan_logic }}</p>
      </div>

      <div class="task-list">
        <div
          v-for="(task, index) in store.hitlState.plan.tasks"
          :key="task.task_id"
          class="task-item"
        >
          <div class="task-status">
            <div class="circle-check">✓</div>
            <div v-if="index !== store.hitlState.plan.tasks.length - 1" class="connecting-line"></div>
          </div>

          <div class="task-content">
            <div class="task-agent">
              <span class="agent-badge">{{ task.agent.replace('_Agent', '') }}</span>
              <span v-if="task.dependency" class="dependency-text">
                (依赖 Task: {{ task.dependency }})
              </span>
            </div>
            <div class="task-action">{{ task.action }}</div>

            <div class="task-args" v-if="task.args.keywords || task.args.target_entity || task.args.time_range">
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

    <div class="card-actions">
      <template v-if="!editMode">
        <button class="btn btn-primary" @click="handleApprove">Run (同意执行)</button>
        <button class="btn btn-text" @click="editMode = true">Deny (修改参数)</button>
      </template>

      <template v-else>
        <textarea
          v-model="localFeedback"
          placeholder="请输入干预指令，例如：'去除对NOAA的分析，增加对美国海军的追踪'..."
          class="feedback-input"
          rows="3"
        ></textarea>
        <div class="edit-actions">
          <button class="btn btn-primary" @click="handleReject">提交重划</button>
          <button class="btn btn-text" @click="editMode = false">取消</button>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.approval-card {
  background-color: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  margin-top: 15px;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  text-align: left;
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
  gap: 8px;
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

/* 任务列表样式 (仿 LightVA 风格) */
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
  max-width: 100%;
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
  justify-content: center; /* 居中按钮 */
}

.btn {
  padding: 6px 20px;
  border: none;
  border-radius: 16px; /* 圆角按钮 */
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-primary { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.btn-primary:hover { background-color: #dcfce7; }
.btn-text { background-color: #ffffff; color: #374151; border: 1px solid #e5e7eb; }
.btn-text:hover { background-color: #f3f4f6; }

.feedback-input {
  width: 100%;
  box-sizing: border-box;
  padding: 8px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-family: inherit;
  margin-bottom: 10px;
  font-size: 13px;
}

.edit-actions { display: flex; gap: 10px; justify-content: flex-end; width: 100%; }
</style>

<!--<script setup>-->
<!--import { useChatStore } from '../../store/chatStore'-->
<!--import { ref, watch } from 'vue'-->

<!--const store = useChatStore()-->
<!--const editMode = ref(false)-->
<!--const localFeedback = ref('')-->

<!--// 当卡片出现时，将默认反馈清空-->
<!--watch(() => store.hitlState.isWaiting, (newVal) => {-->
<!--  if (newVal) localFeedback.value = ''-->
<!--})-->

<!--const handleApprove = () => {-->
<!--  store.hitlState.isWaiting = false-->
<!--  store.messages.push({ id: Date.now(), role: 'user', content: '【系统提示】用户已确认执行该计划。' })-->
<!--  store.isGenerating = true-->
<!--  store.sendFeedback('approve')-->
<!--  setTimeout(() => {-->
<!--    store.messages.push({ id: Date.now() + 1, role: 'ai', content: '计划执行完成！请查看右侧的视图更新。' })-->
<!--    store.isGenerating = false-->
<!--  }, 2000)-->
<!--}-->

<!--const handleReject = () => {-->
<!--  if (!localFeedback.value.trim()) {-->
<!--    alert('请输入修改意见')-->
<!--    return-->
<!--  }-->
<!--  store.hitlState.isWaiting = false-->
<!--  store.messages.push({ id: Date.now(), role: 'user', content: `【修改意见】${localFeedback.value}` })-->
<!--  store.isGenerating = true-->

<!--  // 发送用户的具体修改意见-->
<!--  store.sendFeedback(localFeedback.value)-->
<!--  editMode.value = false-->
<!--}-->
<!--</script>-->

<!--<template>-->
<!--  <div v-if="store.hitlState.isWaiting" class="approval-card">-->
<!--    <div class="card-header">-->
<!--      <span class="icon">⚠️</span> 待审批执行计划-->
<!--    </div>-->

<!--    <div class="card-body">-->
<!--      <div class="plan-item">-->
<!--        <strong>调用探员:</strong>-->
<!--        <span class="tag" v-for="agent in store.hitlState.plan.agents_to_call" :key="agent">-->
<!--          {{ agent.replace('_Agent', '') }}-->
<!--        </span>-->
<!--      </div>-->
<!--      <div class="plan-item">-->
<!--        <strong>分析参数:</strong>-->
<!--        <pre>{{ JSON.stringify(store.hitlState.plan.total_plan_logic, null, 2) }}</pre>-->
<!--      </div>-->
<!--      <div class="plan-item">-->
<!--        <strong>预见图表:</strong>-->
<!--        <span class="tag chart-tag" v-for="chart in store.hitlState.plan.visualizations" :key="chart">-->
<!--          {{ chart }}-->
<!--        </span>-->
<!--      </div>-->
<!--    </div>-->

<!--    <div class="card-actions">-->
<!--      <template v-if="!editMode">-->
<!--        <button class="btn btn-primary" @click="handleApprove">✅ 同意执行</button>-->
<!--        <button class="btn btn-danger" @click="editMode = true">✏️ 修改参数</button>-->
<!--      </template>-->

<!--      <template v-else>-->
<!--        <textarea-->
<!--          v-model="localFeedback"-->
<!--          placeholder="请输入您期望修改的参数或方向，如：'请将时间范围缩小到最近一个月'..."-->
<!--          class="feedback-input"-->
<!--          rows="3"-->
<!--        ></textarea>-->
<!--        <div class="edit-actions">-->
<!--          <button class="btn btn-primary" @click="handleReject">重新规划</button>-->
<!--          <button class="btn btn-text" @click="editMode = false">取消</button>-->
<!--        </div>-->
<!--      </template>-->
<!--    </div>-->
<!--  </div>-->
<!--</template>-->

<!--<style scoped>-->
<!--.approval-card {-->
<!--  background-color: #ffffff;-->
<!--  border: 1px solid #e4e7ed;-->
<!--  border-left: 4px solid #E6A23C; /* 警示色侧边 */-->
<!--  border-radius: 8px;-->
<!--  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);-->
<!--  margin-top: 20px;-->
<!--  overflow: hidden;-->
<!--  animation: slideIn 0.3s ease-out;-->
<!--}-->

<!--@keyframes slideIn {-->
<!--  from { opacity: 0; transform: translateY(10px); }-->
<!--  to { opacity: 1; transform: translateY(0); }-->
<!--}-->

<!--.card-header {-->
<!--  background-color: #fdf6ec;-->
<!--  color: #E6A23C;-->
<!--  padding: 10px 15px;-->
<!--  font-weight: bold;-->
<!--  font-size: 14px;-->
<!--  border-bottom: 1px solid #fbebd4;-->
<!--}-->

<!--.card-body { padding: 15px; font-size: 13px; color: #606266; }-->

<!--.plan-item { margin-bottom: 10px; }-->
<!--.plan-item strong { display: block; margin-bottom: 4px; color: #303133; }-->
<!--.plan-item pre {-->
<!--  background-color: #f4f4f5;-->
<!--  padding: 8px;-->
<!--  border-radius: 4px;-->
<!--  margin: 0;-->
<!--  font-family: monospace;-->
<!--}-->

<!--.tag {-->
<!--  display: inline-block;-->
<!--  background-color: #ecf5ff;-->
<!--  color: #409eff;-->
<!--  padding: 2px 8px;-->
<!--  border-radius: 12px;-->
<!--  margin-right: 6px;-->
<!--  font-size: 12px;-->
<!--}-->
<!--.chart-tag { background-color: #f0f9eb; color: #67c23a; }-->

<!--.card-actions {-->
<!--  padding: 15px;-->
<!--  border-top: 1px solid #ebeef5;-->
<!--  background-color: #fafafa;-->
<!--  display: flex;-->
<!--  gap: 10px;-->
<!--  flex-wrap: wrap;-->
<!--}-->

<!--.btn {-->
<!--  padding: 8px 16px;-->
<!--  border: none;-->
<!--  border-radius: 4px;-->
<!--  cursor: pointer;-->
<!--  font-size: 13px;-->
<!--  transition: all 0.2s;-->
<!--}-->
<!--.btn-primary { background-color: #409EFF; color: white; }-->
<!--.btn-primary:hover { background-color: #66b1ff; }-->
<!--.btn-danger { background-color: #F56C6C; color: white; }-->
<!--.btn-danger:hover { background-color: #f78989; }-->
<!--.btn-text { background: none; color: #909399; }-->
<!--.btn-text:hover { color: #303133; }-->

<!--.feedback-input {-->
<!--  width: 100%;-->
<!--  padding: 8px;-->
<!--  border: 1px solid #dcdfe6;-->
<!--  border-radius: 4px;-->
<!--  resize: vertical;-->
<!--  font-family: inherit;-->
<!--  margin-bottom: 10px;-->
<!--}-->
<!--.edit-actions { display: flex; gap: 10px; width: 100%; }-->
<!--</style>-->