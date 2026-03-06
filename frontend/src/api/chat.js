import { fetchEventSource } from '@microsoft/fetch-event-source'

const BASE_URL = 'http://localhost:5000/api'

// 自定义错误类
class FatalError extends Error {}

async function createSSEConnection(endpoint, payload, callbacks) {
  const ctrl = new AbortController();

  try {
    await fetchEventSource(`${BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal: ctrl.signal,

      // 【终极杀手锏】：即使切换浏览器标签页、最小化窗口，也绝对不中断连接！
      openWhenHidden: true,

      async onopen(response) {
        const contentType = response.headers.get('content-type') || '';
        if (response.ok && contentType.includes('text/event-stream')) {
          return;
        }
        const errText = await response.text();
        throw new FatalError(`Request Error: ${response.status} - ${errText}`);
      },

      onmessage(ev) {
        if (callbacks.onMessage) callbacks.onMessage(ev)

        // 收到业务终止信号，主动物理挂断
        if (ev.event === 'interrupt' || ev.event === 'completed' || ev.event === 'error') {
          ctrl.abort();
        }
      },

      onerror(err) {
        if (callbacks.onError) callbacks.onError(err)
        ctrl.abort(); // 【双重保险】：发生错误时先拔网线
        throw err;
      },

      onclose() {
        if (callbacks.onClose) callbacks.onClose()
        // 【双重保险】：如果后端自然关闭了连接，直接调用 abort 掐死库的重试循环
        ctrl.abort();
      }
    })
  } catch (err) {
    if (err instanceof FatalError || err.name === 'AbortError') {
      console.log('✅ SSE connection terminated safely as expected.');
    } else {
      console.error('SSE stream terminated abnormally:', err);
      // Throw outward for store to catch
      throw err;
    }
  }
}

export const streamChat = (payload, callbacks) => {
  return createSSEConnection('/chat', payload, callbacks)
}

export const streamFeedback = (payload, callbacks) => {
  return createSSEConnection('/feedback', payload, callbacks)
}