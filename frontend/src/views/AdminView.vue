<template>
  <div class="admin-view">
    <el-card>
      <template #header>
        <h3>系统管理 - LLM 配置</h3>
      </template>
      
      <el-form 
        :model="form" 
        :rules="rules" 
        ref="formRef" 
        label-width="120px"
        style="max-width: 600px;"
      >
        <el-form-item label="API 地址" prop="base_url">
          <el-input 
            v-model="form.base_url" 
            placeholder="请输入API地址，例如: https://api.openai.com/v1"
          />
        </el-form-item>
        
        <el-form-item label="API 密钥" prop="api_key">
          <el-input 
            v-model="form.api_key" 
            type="password" 
            placeholder="请输入API密钥"
            show-password
          />
        </el-form-item>
        
        <el-form-item label="模型名称" prop="model">
          <el-input 
            v-model="form.model" 
            placeholder="请输入模型名称，例如: gpt-3.5-turbo"
          />
        </el-form-item>
        
        <el-form-item>
          <el-button 
            type="primary" 
            @click="handleSubmit"
            :loading="loading"
          >
            保存配置
          </el-button>
          <el-button @click="handleReset">
            重置
          </el-button>
        </el-form-item>
      </el-form>
      
      <el-divider />
      
      <div class="config-info">
        <h4>配置说明：</h4>
        <ul>
          <li><strong>API 地址：</strong>大语言模型的API端点地址</li>
          <li><strong>API 密钥：</strong>用于验证API访问权限的密钥</li>
          <li><strong>模型名称：</strong>要使用的具体模型名称</li>
        </ul>
        
        <el-alert
          title="安全提示"
          type="warning"
          :closable="false"
          style="margin-top: 15px;"
        >
          请妥善保管API密钥，不要泄露给他人。配置更改后将立即生效。
        </el-alert>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import apiClient from '../api/axios.js'

const formRef = ref()
const loading = ref(false)

const form = reactive({
  base_url: '',
  api_key: '',
  model: ''
})

const originalForm = reactive({})

const rules = {
  base_url: [
    { required: true, message: '请输入API地址', trigger: 'blur' },
    { type: 'url', message: '请输入正确的URL格式', trigger: 'blur' }
  ],
  api_key: [
    { required: true, message: '请输入API密钥', trigger: 'blur' }
  ],
  model: [
    { required: true, message: '请输入模型名称', trigger: 'blur' }
  ]
}

const fetchConfig = async () => {
  try {
    const response = await apiClient.get('/api/admin/llm-settings')
    
    form.base_url = response.data.base_url || ''
    form.api_key = '' // 不显示已保存的密钥
    form.model = response.data.model || ''
    
    // 保存原始数据用于重置
    Object.assign(originalForm, {
      base_url: form.base_url,
      model: form.model
    })
  } catch (error) {
    if (error.response?.status !== 403) {
      ElMessage.error('获取配置失败')
    }
  }
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (valid) {
      loading.value = true
      
      try {
        await apiClient.put('/api/admin/llm-settings', {
          base_url: form.base_url,
          api_key: form.api_key,
          model: form.model
        })
        
        ElMessage.success('配置保存成功')
        
        // 更新原始数据
        Object.assign(originalForm, {
          base_url: form.base_url,
          model: form.model
        })
        
        // 清空密钥输入框
        form.api_key = ''
      } catch (error) {
        if (error.response?.status === 403) {
          ElMessage.error('权限不足，需要管理员权限')
        } else {
          ElMessage.error(error.response?.data?.detail || '保存失败')
        }
      } finally {
        loading.value = false
      }
    }
  })
}

const handleReset = () => {
  form.base_url = originalForm.base_url
  form.api_key = ''
  form.model = originalForm.model
  
  if (formRef.value) {
    formRef.value.clearValidate()
  }
}

onMounted(() => {
  fetchConfig()
})
</script>

<style scoped>
.admin-view {
  max-width: 800px;
  margin: 0 auto;
}

.config-info {
  background-color: #f8f9fa;
  padding: 20px;
  border-radius: 4px;
}

.config-info h4 {
  margin-top: 0;
  color: #303133;
}

.config-info ul {
  margin: 10px 0;
  padding-left: 20px;
}

.config-info li {
  margin: 8px 0;
  line-height: 1.5;
}
</style>