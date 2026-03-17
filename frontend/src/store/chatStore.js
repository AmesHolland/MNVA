// src/store/chatStore.js
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { streamChat, streamFeedback, geoResolve } from '../api/chat'

export const useChatStore = defineStore('chat', () => {
  // === 1. 状态定义 ===
  const userId = ref('ocean_researcher_01')
  // 尝试从本地读取 session，如果没有则新建
  const sessionId = ref(localStorage.getItem('mnva_session_id') || 'session_' + Date.now())

  // 初始化写入 session
  if (!localStorage.getItem('mnva_session_id')) {
    localStorage.setItem('mnva_session_id', sessionId.value)
  }
  // ==========================================
  // === 🌟 2. 新增：数据仓库状态与持久化 ===
  // ==========================================
  const availableDatasets = ref([])
  // 尝试从本地读取上次选中的数据集
  const activeDatasetId = ref(localStorage.getItem('mnva_active_dataset_id') || '')
  // 🌟 新增：专门用于修改 Dataset ID 的 Action
  const setActiveDataset = (id) => {
    activeDatasetId.value = id
  }
  // 监听数据集切换，自动保存到本地浏览器缓存
  watch(activeDatasetId, (newId) => {
    if (newId) {
      localStorage.setItem('mnva_active_dataset_id', newId)
    } else {
      localStorage.removeItem('mnva_active_dataset_id')
    }
  })
  // 消息结构升级：
  // type: 'text' | 'process' (思维链) | 'plan_card' (审批卡片)
  // content: 文本内容
  // steps: 思维链步骤数组 [{ node: 'intent', status: 'done' }]
  // planData: 审批卡片的数据
  const messages = ref([])

  const isGenerating = ref(false)

  // ==========================================
  // 🛡️ 证据竞争性标注 (Evidence Contestation) 模块
  // ==========================================

  // 1. 核心状态：反驳账本
  // 结构: [{ claim_id: '...', source_id: '...', annotation: 'irrelevant'|'contradicts', reason: '...' }]
  const evidenceAnnotations = ref([])

  // 2. Action: 切换或更新标注状态
  // 交互逻辑：如果用户点同一种按钮，就是取消标注；如果点另一种按钮，就是覆盖标注。
  const toggleEvidenceAnnotation = (claimId, sourceId, annotationType, reason = '') => {
    // 查找是否已经存在对这个 (结论, 证据) 对的标注
    const existingIndex = evidenceAnnotations.value.findIndex(
      (ann) => ann.claim_id === claimId && ann.source_id === sourceId
    )

    if (existingIndex > -1) {
      const existingAnn = evidenceAnnotations.value[existingIndex]
      if (existingAnn.annotation === annotationType) {
        // 情景 A：用户再次点击了相同的状态（例如原本是 irrelevant，又点了一下），视为“撤销打标/恢复默认的 support 状态”
        evidenceAnnotations.value.splice(existingIndex, 1)
      } else {
        // 情景 B：用户切换了态度（例如从 irrelevant 改成了 contradicts），直接覆盖更新
        evidenceAnnotations.value[existingIndex] = {
          claim_id: claimId,
          source_id: sourceId,
          annotation: annotationType,
          reason: reason
        }
      }
    } else {
      // 情景 C：这是一次全新的标注，直接推入账本
      evidenceAnnotations.value.push({
        claim_id: claimId,
        source_id: sourceId,
        annotation: annotationType,
        reason: reason
      })
    }
  }

  // 3. Action: 快速清理特定结论下的所有标注 (可选，用于重置)
  const clearAnnotationsForClaim = (claimId) => {
    evidenceAnnotations.value = evidenceAnnotations.value.filter(ann => ann.claim_id !== claimId)
  }

  // 4. Getter: 获取特定 (结论, 证据) 的当前打标状态，供 UI 高亮按钮使用
  const getAnnotationStatus = (claimId, sourceId) => {
    const ann = evidenceAnnotations.value.find(
      (a) => a.claim_id === claimId && a.source_id === sourceId
    )
    return ann ? ann.annotation : 'support' // 默认状态是 'support' (支持/无异议)
  }

  const analysisResults = ref({
    insight: { title: "Waiting for data...", summary: "Please initiate a query...", keywords: [] },
    visualizations: []
  })
  
  // 【新增】：专门用于存储预览阶段的地图数据
  const previewData = ref(null)
  
  const taskHistory = ref([])

  // 溯源抽屉状态
  // 溯源抽屉状态 (扩展了结论上下文)
  const evidenceState = ref({
    isOpen: false,
    activeSourceIds: [],
    activeClaimId: null,   // 🌟 新增：唤醒抽屉的结论唯一标识
    activeClaimText: ''    // 🌟 新增：唤醒抽屉的结论文本
  })

  // 刷选状态
  const brushState = ref({ timeRange: null, spatialLabels: [] })
  const inputText = ref('Help me analyze the developments regarding deep-sea mining in the United States in the fourth quarter of 2025.')
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
        // console.log(analysisResults.value)
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

  // ==========================================
  // === 🌟 3. 新增：获取数据集列表的方法 ===
  // ==========================================
  const fetchDatasets = async () => {
    try {
      // 这里的 URL 请替换为你后端的实际地址
      const response = await fetch('http://localhost:5000/api/list_datasets')
      const data = await response.json()

      if (data.datasets) {
        availableDatasets.value = data.datasets

        // 智能选择逻辑：
        // 如果当前没有选中的数据集，或者之前选中的数据集被在后台删除了，默认选中最新上传的一个
        const isCurrentValid = availableDatasets.value.some(ds => ds.dataset_id === activeDatasetId.value)

        if ((!activeDatasetId.value || !isCurrentValid) && data.datasets.length > 0) {
          activeDatasetId.value = data.datasets[0].dataset_id // 默认取第一个（最新的）
        }
      }
    } catch (error) {
      console.error('获取数据集列表失败:', error)
    }
  }

  // 初始化加载
  loadFromStorage()

  // === 3. Actions ===

  const openEvidence = (sourceIds, claimId = null, claimText = '') => {
    if (!sourceIds || sourceIds.length === 0) return

    evidenceState.value.activeSourceIds = sourceIds
    evidenceState.value.activeClaimId = claimId      // 绑定上下文
    evidenceState.value.activeClaimText = claimText  // 绑定上下文
    evidenceState.value.isOpen = true
  }

  const closeEvidence = () => {
    evidenceState.value.isOpen = false
    // 延迟清空数据，保证抽屉滑出动画关闭时，里面的内容不会瞬间闪崩消失
    setTimeout(() => {
      evidenceState.value.activeSourceIds = []
      evidenceState.value.activeClaimId = null
      evidenceState.value.activeClaimText = ''
    }, 300)
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
    let promptText = `Please perform a drill-down analysis for the range I have selected:\n`
    if (startTime) promptText += `- Time range: ${startTime} to ${endTime}\n`
    if (labels.length > 0) promptText += `- Spatial/Entity: ${labels.join(', ')}\n`
    promptText += `\nPlease focus on the following key points:`
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
    // 【新增】：处理地图预览事件
    else if (eventName === 'data_profile') {
      console.log("Received map preview data:", data)
      previewData.value = data["preview_map_data"]
      updateBrushState(data["time_range_arr"])
      // 【关键修复】：同时更新 analysisResults，让 SpatiotemporalNavigator 有数据可渲染
      // 我们构造一个虚拟的 Global_Monitor_Agent 任务数据，因为 Navigator 通常读取 geo_dynamic_data
      if (data && data.length > 0) {
        analysisResults.value = {
          // 标记这是一个预览状态，Dashboard 可以据此决定是否显示 Split/Narrative 布局
          isPreview: true, 
          tasks: {
            "preview": {
              agent_name: "Global_Monitor_Agent",
              visualization_data: {
                geo_dynamic_data: data // 将预览数据塞入标准字段
              }
            }
          },
          // 构造一个空的 report 结构，防止报错
          report: null 
        }
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
        // 深度分析完成后，清空预览数据，让主视图接管
        previewData.value = null 
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
    
    // 3. 清空上一轮的预览数据，准备接收新的
    previewData.value = null
    // 同时重置 analysisResults，避免残留上一轮的图表
    // analysisResults.value = { report: null, tasks: {} }
    
    isGenerating.value = true

    try {
      const requestPayload = {
        user_id: userId.value,
        session_id: sessionId.value,
        query: text,
        dataset_id: activeDatasetId.value, // <--- 这里是打通大动脉的关键！
      }
      if (intentOverride) requestPayload.intent_override = intentOverride

      if (evidenceAnnotations.value.length > 0) {
        requestPayload.evidence_annotations = evidenceAnnotations.value
      }
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
    taskHistory, switchTask, previewData,
    evidenceAnnotations,
    toggleEvidenceAnnotation,
    clearAnnotationsForClaim,
    getAnnotationStatus,
    // 暴露新增的
    availableDatasets, activeDatasetId, fetchDatasets, setActiveDataset
  }
})
