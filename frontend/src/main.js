import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css' // 如果不需要默认样式可以删掉这行
import App from './App.vue'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.mount('#app')