<template>
  <div class="recharge-view">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card>
          <template #header>
            <h3>账户充值</h3>
          </template>
          
          <el-form :model="form" label-width="100px">
            <el-form-item label="充值金额">
              <el-input-number 
                v-model="form.amount" 
                :min="1" 
                :max="1000"
                controls-position="right"
                style="width: 200px;"
              />
              <span style="margin-left: 10px;">元</span>
            </el-form-item>
            
            <el-form-item>
              <el-button 
                type="primary" 
                @click="handleRecharge"
                :loading="loading"
              >
                立即充值
              </el-button>
            </el-form-item>
          </el-form>
          
          <el-divider />
          
          <div class="recharge-info">
            <p><strong>充值说明：</strong></p>
            <p>• 1元 = 10点数</p>
            <p>• 翻译一个EPUB文件消耗10点数</p>
            <p>• 充值后点数即时到账</p>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="12">
        <el-card>
          <template #header>
            <h3>充值记录</h3>
          </template>
          
          <el-table :data="orders" style="width: 100%">
            <el-table-column prop="amount" label="金额" width="80">
              <template #default="scope">
                ¥{{ scope.row.amount }}
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="100">
              <template #default="scope">
                <el-tag 
                  :type="scope.row.status === 'completed' ? 'success' : 'warning'"
                >
                  {{ scope.row.status === 'completed' ? '已完成' : '待支付' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="创建时间">
              <template #default="scope">
                {{ formatDate(scope.row.created_at) }}
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 支付二维码弹窗 -->
    <el-dialog v-model="paymentVisible" title="扫码支付" width="400px">
      <div class="payment-dialog">
        <div class="qr-code">
          <img :src="qrCodeUrl" alt="支付二维码" style="width: 200px; height: 200px;" />
        </div>
        <p class="payment-amount">支付金额: ¥{{ currentOrder.amount }}</p>
        <p class="payment-tip">请使用支付宝或微信扫码支付</p>
        
        <div class="payment-status">
          <el-icon v-if="checking" class="is-loading"><Loading /></el-icon>
          <span>{{ checking ? '检查支付状态中...' : '等待支付' }}</span>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading } from '@element-plus/icons-vue'
import apiClient from '../api/axios.js'

const form = reactive({
  amount: 10
})

const loading = ref(false)
const paymentVisible = ref(false)
const qrCodeUrl = ref('')
const currentOrder = ref({})
const checking = ref(false)
const orders = ref([])

let checkTimer = null

const handleRecharge = async () => {
  loading.value = true
  
  try {
    const response = await apiClient.post('/api/recharge', {
      amount: form.amount
    })
    
    currentOrder.value = response.data
    qrCodeUrl.value = response.data.qr_code_url
    paymentVisible.value = true
    
    startPaymentCheck(response.data.order_id)
    
    ElMessage.success('订单创建成功，请扫码支付')
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '创建订单失败')
  } finally {
    loading.value = false
  }
}

const startPaymentCheck = (orderId) => {
  checking.value = true
  
  checkTimer = setInterval(async () => {
    try {
      const response = await apiClient.get(`/api/recharge/status?order_id=${orderId}`)
      
      if (response.data.status === 'completed') {
        clearInterval(checkTimer)
        checking.value = false
        paymentVisible.value = false
        
        ElMessage.success('支付成功！点数已到账')
        await fetchOrders()
      }
    } catch (error) {
      console.error('检查支付状态失败:', error)
    }
  }, 3000)
}

const fetchOrders = async () => {
  try {
    // This endpoint would need to be implemented
    // const response = await apiClient.get('/api/recharge/orders')
    // orders.value = response.data
  } catch (error) {
    console.error('获取充值记录失败:', error)
  }
}

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString('zh-CN')
}

onMounted(() => {
  fetchOrders()
})

// 清理定时器
const cleanup = () => {
  if (checkTimer) {
    clearInterval(checkTimer)
    checkTimer = null
  }
}

// 监听弹窗关闭
const handleDialogClose = () => {
  cleanup()
  checking.value = false
}
</script>

<style scoped>
.recharge-view {
  max-width: 1200px;
  margin: 0 auto;
}

.recharge-info {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.6;
}

.payment-dialog {
  text-align: center;
}

.qr-code {
  margin: 20px 0;
}

.payment-amount {
  font-size: 18px;
  font-weight: bold;
  color: #e6a23c;
  margin: 10px 0;
}

.payment-tip {
  color: #909399;
  margin: 10px 0;
}

.payment-status {
  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
</style>