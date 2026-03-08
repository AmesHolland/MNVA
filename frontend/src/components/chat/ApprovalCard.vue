<!--<script setup>-->
<!--import { useChatStore } from '../../store/chatStore'-->
<!--import { ref, watch } from 'vue'-->

<!--const store = useChatStore()-->
<!--const editMode = ref(false)-->
<!--const localFeedback = ref('')-->

<!--// 当卡片出现时，将默认反馈清空，退出编辑模式-->
<!--watch(() => store.hitlState.isWaiting, (newVal) => {-->
<!--  if (newVal) {-->
<!--    localFeedback.value = ''-->
<!--    editMode.value = false-->
<!--  }-->
<!--})-->

<!--const handleApprove = () => {-->
<!--  // 核心精简：只需调用 store 的方法，无需任何 setTimeout 或额外 push-->
<!--  store.sendFeedback('approve')-->
<!--}-->

<!--const handleReject = () => {-->
<!--  if (!localFeedback.value.trim()) {-->
<!--    alert('Enter revision feedback')-->
<!--    return-->
<!--  }-->
<!--  // 发送修改意见-->
<!--  store.sendFeedback(localFeedback.value)-->
<!--}-->
<!--</script>-->

<!--<template>-->
<!--  <div v-if="store.hitlState.isWaiting && store.hitlState.plan" class="approval-card">-->
<!--    <div class="card-header">-->
<!--      <span class="icon">🍎</span> Task Decomposition-->
<!--    </div>-->

<!--    <div class="card-body">-->
<!--      <div class="plan-summary">-->
<!--        <strong>Plan Logic:</strong>-->
<!--        <p>{{ store.hitlState.plan.total_plan_logic }}</p>-->
<!--      </div>-->

<!--      <div class="task-list">-->
<!--        <div-->
<!--          v-for="(task, index) in store.hitlState.plan.tasks"-->
<!--          :key="task.task_id"-->
<!--          class="task-item"-->
<!--        >-->
<!--          <div class="task-status">-->
<!--            <div class="circle-check">✓</div>-->
<!--            <div v-if="index !== store.hitlState.plan.tasks.length - 1" class="connecting-line"></div>-->
<!--          </div>-->

<!--          <div class="task-content">-->
<!--            <div class="task-agent">-->
<!--              <span class="agent-badge">{{ task.agent.replace('_Agent', '') }}</span>-->
<!--              <span v-if="task.dependency" class="dependency-text">-->
<!--                (Dependency Task: {{ task.dependency }})-->
<!--              </span>-->
<!--            </div>-->
<!--            <div class="task-action">{{ task.action }}</div>-->

<!--            <div class="task-args" v-if="task.args.keywords || task.args.target_entity || task.args.time_range">-->
<!--              <span class="arg-tag" v-if="task.args.keywords">-->
<!--                🗝️ {{ task.args.keywords }}-->
<!--              </span>-->
<!--              <span class="arg-tag" v-if="task.args.target_entity">-->
<!--                🎯 {{ task.args.target_entity }}-->
<!--              </span>-->
<!--              <span class="arg-tag" v-if="task.args.time_range">-->
<!--                ⏳ {{ task.args.time_range }}-->
<!--              </span>-->
<!--            </div>-->
<!--          </div>-->
<!--        </div>-->
<!--      </div>-->
<!--    </div>-->

<!--    <div class="card-actions">-->
<!--      <template v-if="!editMode">-->
<!--        <button class="btn btn-primary" @click="handleApprove">Run </button>-->
<!--        <button class="btn btn-text" @click="editMode = true">Deny </button>-->
<!--      </template>-->

<!--      <template v-else>-->
<!--        <textarea-->
<!--          v-model="localFeedback"-->
<!--          placeholder="Please enter intervention instructions, e.g.: 'Remove the analysis of NOAA and add tracking of the U.S. Navy'..."-->
<!--          class="feedback-input"-->
<!--          rows="3"-->
<!--        ></textarea>-->
<!--        <div class="edit-actions">-->
<!--            <button class="btn btn-primary" @click="handleReject">Submit Replan</button>-->
<!--            <button class="btn btn-text" @click="editMode = false">Cancel</button>-->
<!--        </div>-->
<!--      </template>-->
<!--    </div>-->
<!--  </div>-->
<!--</template>-->

<!--<style scoped>-->
<!--.approval-card {-->
<!--  background-color: #ffffff;-->
<!--  border: 1px solid #e4e7ed;-->
<!--  border-radius: 8px;-->
<!--  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);-->
<!--  margin-top: 15px;-->
<!--  overflow: hidden;-->
<!--  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;-->
<!--  text-align: left;-->
<!--}-->

<!--.card-header {-->
<!--  background-color: #fafafa;-->
<!--  color: #303133;-->
<!--  padding: 12px 15px;-->
<!--  font-weight: 600;-->
<!--  font-size: 14px;-->
<!--  border-bottom: 1px solid #ebeef5;-->
<!--  display: flex;-->
<!--  align-items: center;-->
<!--  gap: 8px;-->
<!--}-->

<!--.card-body {-->
<!--  padding: 15px;-->
<!--}-->

<!--/* 规划总览样式 */-->
<!--.plan-summary {-->
<!--  background-color: #f3f4f6;-->
<!--  padding: 10px;-->
<!--  border-radius: 6px;-->
<!--  margin-bottom: 15px;-->
<!--  font-size: 13px;-->
<!--  color: #606266;-->
<!--  line-height: 1.5;-->
<!--}-->
<!--.plan-summary strong {-->
<!--  color: #303133;-->
<!--  display: block;-->
<!--  margin-bottom: 4px;-->
<!--}-->
<!--.plan-summary p {-->
<!--  margin: 0;-->
<!--}-->

<!--/* 任务列表样式 (仿 LightVA 风格) */-->
<!--.task-list {-->
<!--  display: flex;-->
<!--  flex-direction: column;-->
<!--}-->

<!--.task-item {-->
<!--  display: flex;-->
<!--  gap: 12px;-->
<!--  margin-bottom: 12px;-->
<!--}-->

<!--.task-status {-->
<!--  display: flex;-->
<!--  flex-direction: column;-->
<!--  align-items: center;-->
<!--  width: 20px;-->
<!--}-->

<!--.circle-check {-->
<!--  width: 18px;-->
<!--  height: 18px;-->
<!--  background-color: #333;-->
<!--  color: white;-->
<!--  border-radius: 50%;-->
<!--  display: flex;-->
<!--  align-items: center;-->
<!--  justify-content: center;-->
<!--  font-size: 10px;-->
<!--  z-index: 2;-->
<!--}-->

<!--.connecting-line {-->
<!--  width: 2px;-->
<!--  flex-grow: 1;-->
<!--  background-color: #e4e7ed;-->
<!--  margin-top: 4px;-->
<!--  min-height: 20px;-->
<!--}-->

<!--.task-content {-->
<!--  background-color: #f9fafc;-->
<!--  border: 1px solid #ebeef5;-->
<!--  border-radius: 6px;-->
<!--  padding: 10px;-->
<!--  flex-grow: 1;-->
<!--}-->

<!--.task-agent {-->
<!--  margin-bottom: 6px;-->
<!--}-->

<!--.agent-badge {-->
<!--  background-color: #409EFF;-->
<!--  color: white;-->
<!--  font-size: 12px;-->
<!--  padding: 2px 6px;-->
<!--  border-radius: 4px;-->
<!--  font-weight: 500;-->
<!--}-->

<!--.dependency-text {-->
<!--  font-size: 12px;-->
<!--  color: #909399;-->
<!--  margin-left: 8px;-->
<!--}-->

<!--.task-action {-->
<!--  font-size: 13px;-->
<!--  color: #303133;-->
<!--  line-height: 1.4;-->
<!--  margin-bottom: 8px;-->
<!--}-->

<!--.task-args {-->
<!--  display: flex;-->
<!--  flex-wrap: wrap;-->
<!--  gap: 6px;-->
<!--}-->

<!--.arg-tag {-->
<!--  background-color: #ecf5ff;-->
<!--  color: #409eff;-->
<!--  border: 1px solid #b3d8ff;-->
<!--  font-size: 11px;-->
<!--  padding: 2px 6px;-->
<!--  border-radius: 4px;-->
<!--  max-width: 100%;-->
<!--  overflow: hidden;-->
<!--  text-overflow: ellipsis;-->
<!--  white-space: nowrap;-->
<!--}-->

<!--/* 底部按钮区 */-->
<!--.card-actions {-->
<!--  padding: 12px 15px;-->
<!--  border-top: 1px solid #ebeef5;-->
<!--  background-color: #fafafa;-->
<!--  display: flex;-->
<!--  gap: 10px;-->
<!--  justify-content: center; /* 居中按钮 */-->
<!--}-->

<!--.btn {-->
<!--  padding: 6px 20px;-->
<!--  border: none;-->
<!--  border-radius: 16px; /* 圆角按钮 */-->
<!--  cursor: pointer;-->
<!--  font-size: 13px;-->
<!--  font-weight: 500;-->
<!--  transition: all 0.2s;-->
<!--}-->

<!--.btn-primary { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }-->
<!--.btn-primary:hover { background-color: #dcfce7; }-->
<!--.btn-text { background-color: #ffffff; color: #374151; border: 1px solid #e5e7eb; }-->
<!--.btn-text:hover { background-color: #f3f4f6; }-->

<!--.feedback-input {-->
<!--  width: 100%;-->
<!--  box-sizing: border-box;-->
<!--  padding: 8px;-->
<!--  border: 1px solid #dcdfe6;-->
<!--  border-radius: 4px;-->
<!--  font-family: inherit;-->
<!--  margin-bottom: 10px;-->
<!--  font-size: 13px;-->
<!--}-->

<!--.edit-actions { display: flex; gap: 10px; justify-content: flex-end; width: 100%; }-->
<!--</style>-->

<!--&lt;!&ndash;<script setup>&ndash;&gt;-->
<!--&lt;!&ndash;import { useChatStore } from '../../store/chatStore'&ndash;&gt;-->
<!--&lt;!&ndash;import { ref, watch } from 'vue'&ndash;&gt;-->

<!--&lt;!&ndash;const store = useChatStore()&ndash;&gt;-->
<!--&lt;!&ndash;const editMode = ref(false)&ndash;&gt;-->
<!--&lt;!&ndash;const localFeedback = ref('')&ndash;&gt;-->

<!--&lt;!&ndash;// 当卡片出现时，将默认反馈清空&ndash;&gt;-->
<!--&lt;!&ndash;watch(() => store.hitlState.isWaiting, (newVal) => {&ndash;&gt;-->
<!--&lt;!&ndash;  if (newVal) localFeedback.value = ''&ndash;&gt;-->
<!--&lt;!&ndash;})&ndash;&gt;-->

<!--&lt;!&ndash;const handleApprove = () => {&ndash;&gt;-->
<!--&lt;!&ndash;  store.hitlState.isWaiting = false&ndash;&gt;-->
<!--&lt;!&ndash;  store.messages.push({ id: Date.now(), role: 'user', content: '【系统提示】用户已确认执行该计划。' })&ndash;&gt;-->
<!--&lt;!&ndash;  store.isGenerating = true&ndash;&gt;-->
<!--&lt;!&ndash;  store.sendFeedback('approve')&ndash;&gt;-->
<!--&lt;!&ndash;  setTimeout(() => {&ndash;&gt;-->
<!--&lt;!&ndash;    store.messages.push({ id: Date.now() + 1, role: 'ai', content: '计划执行完成！请查看右侧的视图更新。' })&ndash;&gt;-->
<!--&lt;!&ndash;    store.isGenerating = false&ndash;&gt;-->
<!--&lt;!&ndash;  }, 2000)&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;const handleReject = () => {&ndash;&gt;-->
<!--&lt;!&ndash;  if (!localFeedback.value.trim()) {&ndash;&gt;-->
<!--&lt;!&ndash;    alert('请输入修改意见')&ndash;&gt;-->
<!--&lt;!&ndash;    return&ndash;&gt;-->
<!--&lt;!&ndash;  }&ndash;&gt;-->
<!--&lt;!&ndash;  store.hitlState.isWaiting = false&ndash;&gt;-->
<!--&lt;!&ndash;  store.messages.push({ id: Date.now(), role: 'user', content: `【修改意见】${localFeedback.value}` })&ndash;&gt;-->
<!--&lt;!&ndash;  store.isGenerating = true&ndash;&gt;-->

<!--&lt;!&ndash;  // 发送用户的具体修改意见&ndash;&gt;-->
<!--&lt;!&ndash;  store.sendFeedback(localFeedback.value)&ndash;&gt;-->
<!--&lt;!&ndash;  editMode.value = false&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->
<!--&lt;!&ndash;</script>&ndash;&gt;-->

<!--&lt;!&ndash;<template>&ndash;&gt;-->
<!--&lt;!&ndash;  <div v-if="store.hitlState.isWaiting" class="approval-card">&ndash;&gt;-->
<!--&lt;!&ndash;    <div class="card-header">&ndash;&gt;-->
<!--&lt;!&ndash;      <span class="icon">⚠️</span> 待审批执行计划&ndash;&gt;-->
<!--&lt;!&ndash;    </div>&ndash;&gt;-->

<!--&lt;!&ndash;    <div class="card-body">&ndash;&gt;-->
<!--&lt;!&ndash;      <div class="plan-item">&ndash;&gt;-->
<!--&lt;!&ndash;        <strong>调用探员:</strong>&ndash;&gt;-->
<!--&lt;!&ndash;        <span class="tag" v-for="agent in store.hitlState.plan.agents_to_call" :key="agent">&ndash;&gt;-->
<!--&lt;!&ndash;          {{ agent.replace('_Agent', '') }}&ndash;&gt;-->
<!--&lt;!&ndash;        </span>&ndash;&gt;-->
<!--&lt;!&ndash;      </div>&ndash;&gt;-->
<!--&lt;!&ndash;      <div class="plan-item">&ndash;&gt;-->
<!--&lt;!&ndash;        <strong>分析参数:</strong>&ndash;&gt;-->
<!--&lt;!&ndash;        <pre>{{ JSON.stringify(store.hitlState.plan.total_plan_logic, null, 2) }}</pre>&ndash;&gt;-->
<!--&lt;!&ndash;      </div>&ndash;&gt;-->
<!--&lt;!&ndash;      <div class="plan-item">&ndash;&gt;-->
<!--&lt;!&ndash;        <strong>预见图表:</strong>&ndash;&gt;-->
<!--&lt;!&ndash;        <span class="tag chart-tag" v-for="chart in store.hitlState.plan.visualizations" :key="chart">&ndash;&gt;-->
<!--&lt;!&ndash;          {{ chart }}&ndash;&gt;-->
<!--&lt;!&ndash;        </span>&ndash;&gt;-->
<!--&lt;!&ndash;      </div>&ndash;&gt;-->
<!--&lt;!&ndash;    </div>&ndash;&gt;-->

<!--&lt;!&ndash;    <div class="card-actions">&ndash;&gt;-->
<!--&lt;!&ndash;      <template v-if="!editMode">&ndash;&gt;-->
<!--&lt;!&ndash;        <button class="btn btn-primary" @click="handleApprove">✅ 同意执行</button>&ndash;&gt;-->
<!--&lt;!&ndash;        <button class="btn btn-danger" @click="editMode = true">✏️ 修改参数</button>&ndash;&gt;-->
<!--&lt;!&ndash;      </template>&ndash;&gt;-->

<!--&lt;!&ndash;      <template v-else>&ndash;&gt;-->
<!--&lt;!&ndash;        <textarea&ndash;&gt;-->
<!--&lt;!&ndash;          v-model="localFeedback"&ndash;&gt;-->
<!--&lt;!&ndash;          placeholder="请输入您期望修改的参数或方向，如：'请将时间范围缩小到最近一个月'..."&ndash;&gt;-->
<!--&lt;!&ndash;          class="feedback-input"&ndash;&gt;-->
<!--&lt;!&ndash;          rows="3"&ndash;&gt;-->
<!--&lt;!&ndash;        ></textarea>&ndash;&gt;-->
<!--&lt;!&ndash;        <div class="edit-actions">&ndash;&gt;-->
<!--&lt;!&ndash;          <button class="btn btn-primary" @click="handleReject">重新规划</button>&ndash;&gt;-->
<!--&lt;!&ndash;          <button class="btn btn-text" @click="editMode = false">取消</button>&ndash;&gt;-->
<!--&lt;!&ndash;        </div>&ndash;&gt;-->
<!--&lt;!&ndash;      </template>&ndash;&gt;-->
<!--&lt;!&ndash;    </div>&ndash;&gt;-->
<!--&lt;!&ndash;  </div>&ndash;&gt;-->
<!--&lt;!&ndash;</template>&ndash;&gt;-->

<!--&lt;!&ndash;<style scoped>&ndash;&gt;-->
<!--&lt;!&ndash;.approval-card {&ndash;&gt;-->
<!--&lt;!&ndash;  background-color: #ffffff;&ndash;&gt;-->
<!--&lt;!&ndash;  border: 1px solid #e4e7ed;&ndash;&gt;-->
<!--&lt;!&ndash;  border-left: 4px solid #E6A23C; /* 警示色侧边 */&ndash;&gt;-->
<!--&lt;!&ndash;  border-radius: 8px;&ndash;&gt;-->
<!--&lt;!&ndash;  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.05);&ndash;&gt;-->
<!--&lt;!&ndash;  margin-top: 20px;&ndash;&gt;-->
<!--&lt;!&ndash;  overflow: hidden;&ndash;&gt;-->
<!--&lt;!&ndash;  animation: slideIn 0.3s ease-out;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;@keyframes slideIn {&ndash;&gt;-->
<!--&lt;!&ndash;  from { opacity: 0; transform: translateY(10px); }&ndash;&gt;-->
<!--&lt;!&ndash;  to { opacity: 1; transform: translateY(0); }&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;.card-header {&ndash;&gt;-->
<!--&lt;!&ndash;  background-color: #fdf6ec;&ndash;&gt;-->
<!--&lt;!&ndash;  color: #E6A23C;&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 10px 15px;&ndash;&gt;-->
<!--&lt;!&ndash;  font-weight: bold;&ndash;&gt;-->
<!--&lt;!&ndash;  font-size: 14px;&ndash;&gt;-->
<!--&lt;!&ndash;  border-bottom: 1px solid #fbebd4;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;.card-body { padding: 15px; font-size: 13px; color: #606266; }&ndash;&gt;-->

<!--&lt;!&ndash;.plan-item { margin-bottom: 10px; }&ndash;&gt;-->
<!--&lt;!&ndash;.plan-item strong { display: block; margin-bottom: 4px; color: #303133; }&ndash;&gt;-->
<!--&lt;!&ndash;.plan-item pre {&ndash;&gt;-->
<!--&lt;!&ndash;  background-color: #f4f4f5;&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 8px;&ndash;&gt;-->
<!--&lt;!&ndash;  border-radius: 4px;&ndash;&gt;-->
<!--&lt;!&ndash;  margin: 0;&ndash;&gt;-->
<!--&lt;!&ndash;  font-family: monospace;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;.tag {&ndash;&gt;-->
<!--&lt;!&ndash;  display: inline-block;&ndash;&gt;-->
<!--&lt;!&ndash;  background-color: #ecf5ff;&ndash;&gt;-->
<!--&lt;!&ndash;  color: #409eff;&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 2px 8px;&ndash;&gt;-->
<!--&lt;!&ndash;  border-radius: 12px;&ndash;&gt;-->
<!--&lt;!&ndash;  margin-right: 6px;&ndash;&gt;-->
<!--&lt;!&ndash;  font-size: 12px;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->
<!--&lt;!&ndash;.chart-tag { background-color: #f0f9eb; color: #67c23a; }&ndash;&gt;-->

<!--&lt;!&ndash;.card-actions {&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 15px;&ndash;&gt;-->
<!--&lt;!&ndash;  border-top: 1px solid #ebeef5;&ndash;&gt;-->
<!--&lt;!&ndash;  background-color: #fafafa;&ndash;&gt;-->
<!--&lt;!&ndash;  display: flex;&ndash;&gt;-->
<!--&lt;!&ndash;  gap: 10px;&ndash;&gt;-->
<!--&lt;!&ndash;  flex-wrap: wrap;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->

<!--&lt;!&ndash;.btn {&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 8px 16px;&ndash;&gt;-->
<!--&lt;!&ndash;  border: none;&ndash;&gt;-->
<!--&lt;!&ndash;  border-radius: 4px;&ndash;&gt;-->
<!--&lt;!&ndash;  cursor: pointer;&ndash;&gt;-->
<!--&lt;!&ndash;  font-size: 13px;&ndash;&gt;-->
<!--&lt;!&ndash;  transition: all 0.2s;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->
<!--&lt;!&ndash;.btn-primary { background-color: #409EFF; color: white; }&ndash;&gt;-->
<!--&lt;!&ndash;.btn-primary:hover { background-color: #66b1ff; }&ndash;&gt;-->
<!--&lt;!&ndash;.btn-danger { background-color: #F56C6C; color: white; }&ndash;&gt;-->
<!--&lt;!&ndash;.btn-danger:hover { background-color: #f78989; }&ndash;&gt;-->
<!--&lt;!&ndash;.btn-text { background: none; color: #909399; }&ndash;&gt;-->
<!--&lt;!&ndash;.btn-text:hover { color: #303133; }&ndash;&gt;-->

<!--&lt;!&ndash;.feedback-input {&ndash;&gt;-->
<!--&lt;!&ndash;  width: 100%;&ndash;&gt;-->
<!--&lt;!&ndash;  padding: 8px;&ndash;&gt;-->
<!--&lt;!&ndash;  border: 1px solid #dcdfe6;&ndash;&gt;-->
<!--&lt;!&ndash;  border-radius: 4px;&ndash;&gt;-->
<!--&lt;!&ndash;  resize: vertical;&ndash;&gt;-->
<!--&lt;!&ndash;  font-family: inherit;&ndash;&gt;-->
<!--&lt;!&ndash;  margin-bottom: 10px;&ndash;&gt;-->
<!--&lt;!&ndash;}&ndash;&gt;-->
<!--&lt;!&ndash;.edit-actions { display: flex; gap: 10px; width: 100%; }&ndash;&gt;-->
<!--&lt;!&ndash;</style>&ndash;&gt;-->