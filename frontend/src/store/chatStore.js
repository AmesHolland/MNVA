// src/store/chatStore.js
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { streamChat, streamFeedback, geoResolve } from '../api/chat'

export const useChatStore = defineStore('chat', () => {
  // === 1. 状态定义 ===
  const userId = ref('ocean_researcher_01')
  // 尝试从本地读取 session，如果没有则新建
  const sessionId = ref(localStorage.getItem('mnva_session_id') || 'session_' + Date.now())

  // 消息结构升级：
  // type: 'text' | 'process' (思维链) | 'plan_card' (审批卡片)
  // content: 文本内容
  // steps: 思维链步骤数组 [{ node: 'intent', status: 'done' }]
  // planData: 审批卡片的数据
  const messages = ref([])

  const isGenerating = ref(false)
  
  // 移除旧的 hitlState，改为消息驱动
  // const hitlState = ref(...) 

  const analysisResults = ref({
    insight: { title: "Waiting for data...", summary: "Please initiate a query...", keywords: [] },
    visualizations: []
  })
  
  const taskHistory = ref([])

  // 溯源抽屉状态
  const evidenceState = ref({ isOpen: false, activeSourceIds: [] })

  // 刷选状态
  const brushState = ref({ timeRange: null, spatialLabels: [] })
  const inputText = ref('帮我分析2025年第四季度美国在深海采矿方面的动态')
  const pendingSandboxContext = ref(null)

  // === 2. 持久化逻辑 (解决问题 1) ===
  const loadFromStorage = () => {
    const saved = localStorage.getItem('mnva_state')
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        // 恢复关键状态
        if (parsed.messages) messages.value = parsed.messages
        if (parsed.analysisResults) analysisResults.value = parsed.analysisResults
        if (parsed.taskHistory) taskHistory.value = parsed.taskHistory
        // 恢复 sessionId 确保后端能接上上下文
        if (parsed.sessionId) sessionId.value = parsed.sessionId
      } catch (e) {
        console.error('Failed to load state', e)
      }
    }
  }

  // 监听状态变化并保存
  watch(
    [messages, analysisResults, taskHistory, sessionId],
    () => {
      const stateToSave = {
        messages: messages.value,
        analysisResults: analysisResults.value,
        taskHistory: taskHistory.value,
        sessionId: sessionId.value
      }
      localStorage.setItem('mnva_state', JSON.stringify(stateToSave))
      localStorage.setItem('mnva_session_id', sessionId.value)
    },
    { deep: true }
  )

  // 初始化加载
  loadFromStorage()

  // === 3. Actions ===

  const openEvidence = (sourceIds) => {
    if (!sourceIds || sourceIds.length === 0) return
    evidenceState.value.activeSourceIds = sourceIds
    evidenceState.value.isOpen = true
  }

  const closeEvidence = () => {
    evidenceState.value.isOpen = false
    setTimeout(() => { evidenceState.value.activeSourceIds = [] }, 300)
  }

  const updateBrushState = (timeRange, spatialLabels = null) => {
    brushState.value.timeRange = timeRange
    if (spatialLabels !== null) brushState.value.spatialLabels = spatialLabels
  }
  
  // 【新增】：处理地图框选，调用后端反解坐标
  const resolveMapBrush = async (coordinates) => {
    if (!coordinates || coordinates.length === 0) return
    
    // 先清空之前的标签，或者保留？这里选择覆盖
    // brushState.value.spatialLabels = ['Resolving...'] 
    
    const result = await geoResolve(coordinates)
    if (result && result.regions) {
      // 去重合并
      const newLabels = [...new Set([...brushState.value.spatialLabels, ...result.regions])]
      brushState.value.spatialLabels = newLabels
    }
  }

  const clearBrushState = () => {
    brushState.value.timeRange = null
    brushState.value.spatialLabels = []
  }

  const formatDate = (timestamp) => {
    if (!timestamp) return ''
    const d = new Date(timestamp)
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  }

  const prepareSandboxAnalysis = () => {
    const timeRange = brushState.value.timeRange
    const labels = brushState.value.spatialLabels || []
    if (!timeRange && labels.length === 0) return

    let startTime = null, endTime = null
    if (timeRange) {
      const d1 = new Date(timeRange[0]); startTime = `${d1.getFullYear()}-${String(d1.getMonth()+1).padStart(2,'0')}-${String(d1.getDate()).padStart(2,'0')}`
      const d2 = new Date(timeRange[1]); endTime = `${d2.getFullYear()}-${String(d2.getMonth()+1).padStart(2,'0')}-${String(d2.getDate()).padStart(2,'0')}`
    }

    let promptText = `请针对我圈选的范围进行下钻分析：\n`
    if (startTime) promptText += `- 时间范围：${startTime} 至 ${endTime}\n`
    if (labels.length > 0) promptText += `- 空间/实体：${labels.join(', ')}\n`
    promptText += `\n请重点关注：`

    inputText.value = promptText
    pendingSandboxContext.value = {
      is_sandbox_request: true,
      sandbox_constraints: { start_time: startTime, end_time: endTime, spatial_labels: labels }
    }
  }
  
  const switchTask = (task) => {
    if (task && task.results) {
      analysisResults.value = task.results
    }
  }

  // === 4. 核心：SSE 消息处理 (解决问题 2 & 3) ===
  const handleSSEMessage = (ev) => {
    const eventName = ev.event
    const data = JSON.parse(ev.data)
    
    // 获取最后一条消息（通常是 AI 正在生成的那条）
    let lastMsg = messages.value[messages.value.length - 1]

    // 如果最后一条不是 AI 消息，或者已经结束了（比如上一轮对话），则新建一条
    // 注意：我们在 sendMessage 时已经预置了一条 'process' 类型的消息

    if (eventName === 'node_progress') {
      // 确保当前有一个 "process" 类型的消息在最底部
      if (!lastMsg || lastMsg.role !== 'ai' || lastMsg.type !== 'process') {
         // 理论上 sendMessage 会创建，但为了健壮性
         return 
      }
      
      // 往 steps 数组里追加节点状态
      // 去重：如果最后一个 step 就是当前 node，就不加了
      const lastStep = lastMsg.steps[lastMsg.steps.length - 1]
      if (!lastStep || lastStep.node !== data.node) {
        lastMsg.steps.push({
          node: data.node,
          timestamp: Date.now(),
          status: 'loading' // 刚收到是 loading，收到下一个时把上一个置为 done? 简化起见，收到即视为进入该节点
        })
      }
    }
    else if (eventName === 'interrupt') {
      // 1. 结束当前的 Process 消息流
      isGenerating.value = false
      
      // 2. 插入一条 "审批卡片" 消息
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        type: 'plan_card',
        content: 'Plan Generated',
        planData: data.plan,
        isApproved: false // 标记是否已点击
      })
    }
    else if (eventName === 'completed') {
      isGenerating.value = false
      
      // 更新历史
      if (data.task_history) taskHistory.value = data.task_history

      // 处理可视化结果
      if (data.is_new_visual_result) {
        analysisResults.value = data.results
      }
      
      // 插入最终回答文本
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        type: 'text',
        content: data.direct_answer || 'Analysis completed.'
      })
    }
    else if (eventName === 'error') {
      isGenerating.value = false
      messages.value.push({
        id: Date.now(),
        role: 'ai',
        type: 'text',
        content: `【System Error】: ${data.error}`
      })
    }
  }

  const sendMessage = async (text, intentOverride = null) => {
    if (!text.trim() || isGenerating.value) return

    // 1. 用户消息
    messages.value.push({ id: Date.now(), role: 'user', type: 'text', content: text })
    
    // 2. 预置 AI 的 "思考过程" 消息 (Process Log)
    messages.value.push({ 
      id: Date.now() + 1, 
      role: 'ai', 
      type: 'process', 
      content: 'Thinking...', 
      steps: [], // 存放节点流
      isExpanded: true // 默认展开，或者 false
    })
    
    isGenerating.value = true

    try {
      const requestPayload = {
        user_id: userId.value,
        session_id: sessionId.value,
        query: text
      }
      if (intentOverride) requestPayload.intent_override = intentOverride

      await streamChat(requestPayload, {
        onMessage: handleSSEMessage,
        onError: (err) => {
          console.error('SSE Error:', err)
          isGenerating.value = false
          messages.value.push({ id: Date.now(), role: 'ai', type: 'text', content: `Error: ${err.message}` })
        }
      })
    } catch (error) {
      console.error('Failed to initiate request', error)
      isGenerating.value = false
    }
  }

  const sendFeedback = async (feedbackText, messageId) => {
    // 找到对应的 plan_card 消息，将其标记为已处理
    const msgIndex = messages.value.findIndex(m => m.id === messageId)
    if (msgIndex !== -1) {
      messages.value[msgIndex].isApproved = true
    }

    // 用户反馈消息
    const isApprove = feedbackText === 'approve'
    messages.value.push({
      id: Date.now(),
      role: 'user',
      type: 'text',
      content: isApprove ? '✅ Plan Approved' : `Modification: ${feedbackText}`
    })

    // 再次预置 AI 的 "思考过程" 消息 (第二阶段执行)
    messages.value.push({ 
      id: Date.now() + 1, 
      role: 'ai', 
      type: 'process', 
      content: 'Executing Analysis...', 
      steps: [], 
      isExpanded: true 
    })

    isGenerating.value = true

    try {
      await streamFeedback({
        user_id: userId.value,
        session_id: sessionId.value,
        feedback: feedbackText
      }, {
        onMessage: handleSSEMessage,
        onError: (err) => {
          console.error('SSE Error:', err)
          isGenerating.value = false
        }
      })
    } catch (error) {
      console.error('Failed to send feedback', error)
      isGenerating.value = false
    }
  }

  return { 
    sessionId, messages, isGenerating, analysisResults, sendMessage, sendFeedback, 
    evidenceState, openEvidence, closeEvidence,
    brushState, updateBrushState, clearBrushState, resolveMapBrush,
    formatDate, inputText, pendingSandboxContext, prepareSandboxAnalysis,
    taskHistory, switchTask
  }
})
