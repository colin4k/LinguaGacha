<template>
  <div class="translation-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <template #header>
            <h3>EPUB 文件翻译</h3>
          </template>
          
          <el-form :model="form" label-width="120px">
            <el-form-item label="源语言">
              <el-select v-model="form.sourceLanguage" placeholder="请选择源语言">
                <el-option label="自动检测" value="auto" />
                <el-option label="英语" value="en" />
                <el-option label="中文" value="zh" />
                <el-option label="日语" value="ja" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="目标语言">
              <el-select v-model="form.targetLanguage" placeholder="请选择目标语言">
                <el-option label="中文" value="zh" />
                <el-option label="英语" value="en" />
                <el-option label="日语" value="ja" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="上传文件">
              <el-upload
                ref="uploadRef"
                :auto-upload="false"
                :on-change="handleFileChange"
                :file-list="fileList"
                accept=".epub"
                :limit="1"
              >
                <el-button type="primary">选择 EPUB 文件</el-button>
                <template #tip>
                  <div class="el-upload__tip">
                    只能上传 EPUB 文件，且不超过 50MB
                  </div>
                </template>
              </el-upload>
            </el-form-item>
            
            <el-form-item>
              <el-button 
                type="success" 
                @click="handleUpload" 
                :loading="uploading"
                :disabled="!selectedFile"
              >
                开始翻译
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>
    </el-row>
    
    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card>
          <template #header>
            <h3>翻译任务列表</h3>
          </template>
          
          <el-table :data="tasks" style="width: 100%">
            <el-table-column prop="original_filename" label="文件名" />
            <el-table-column prop="source_language" label="源语言" />
            <el-table-column prop="target_language" label="目标语言" />
            <el-table-column prop="status" label="状态">
              <template #default="scope">
                <el-tag 
                  :type="getStatusType(scope.row.status)"
                >
                  {{ getStatusText(scope.row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="创建时间">
              <template #default="scope">
                {{ formatDate(scope.row.created_at) }}
              </template>
            </el-table-column>
            <el-table-column label="操作">
              <template #default="scope">
                <el-button
                  v-if="scope.row.status === 'completed'"
                  type="primary"
                  size="small"
                  @click="handleDownload(scope.row.id)"
                >
                  下载
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import apiClient from '../api/axios.js'
import { supabase } from '../supabase.js'
import { useAuthStore } from '../stores/auth.js'

const authStore = useAuthStore()

const form = reactive({
  sourceLanguage: 'auto',
  targetLanguage: 'zh'
})

const fileList = ref([])
const selectedFile = ref(null)
const uploading = ref(false)
const tasks = ref([])

let subscription = null

const handleFileChange = (file) => {
  selectedFile.value = file.raw
  fileList.value = [file]
}

const handleUpload = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请选择文件')
    return
  }
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  formData.append('source_language', form.sourceLanguage)
  formData.append('target_language', form.targetLanguage)
  
  uploading.value = true
  
  try {
    await apiClient.post('/api/translations/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    ElMessage.success('文件上传成功，开始翻译')
    fileList.value = []
    selectedFile.value = null
    await fetchTasks()
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '上传失败')
  } finally {
    uploading.value = false
  }
}

const fetchTasks = async () => {
  try {
    const response = await apiClient.get('/api/translations')
    tasks.value = response.data
  } catch (error) {
    ElMessage.error('获取任务列表失败')
  }
}

const handleDownload = async (taskId) => {
  try {
    const response = await apiClient.get(`/api/translations/${taskId}/download`, {
      responseType: 'blob'
    })
    
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    
    const contentDisposition = response.headers['content-disposition']
    let fileName = 'translated_file.epub'
    if (contentDisposition) {
      const fileNameMatch = contentDisposition.match(/filename="(.+)"/)
      if (fileNameMatch) {
        fileName = fileNameMatch[1]
      }
    }
    
    link.setAttribute('download', fileName)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('下载成功')
  } catch (error) {
    ElMessage.error('下载失败')
  }
}

const getStatusType = (status) => {
  const statusMap = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return statusMap[status] || 'info'
}

const getStatusText = (status) => {
  const statusMap = {
    pending: '等待中',
    processing: '翻译中',
    completed: '已完成',
    failed: '失败'
  }
  return statusMap[status] || status
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString('zh-CN')
}

const setupRealtimeSubscription = () => {
  if (!authStore.user) return
  
  subscription = supabase
    .channel('translation_tasks')
    .on('postgres_changes', 
      { 
        event: '*', 
        schema: 'public', 
        table: 'translation_tasks',
        filter: `user_id=eq.${authStore.user.id}`
      }, 
      () => {
        fetchTasks()
      }
    )
    .subscribe()
}

onMounted(() => {
  fetchTasks()
  setupRealtimeSubscription()
})

onUnmounted(() => {
  if (subscription) {
    supabase.removeChannel(subscription)
  }
})
</script>

<style scoped>
.translation-view {
  max-width: 1200px;
  margin: 0 auto;
}
</style>