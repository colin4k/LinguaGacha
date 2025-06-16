<template>
  <el-container class="dashboard-container">
    <el-aside width="200px">
      <el-menu
        :default-active="$route.path"
        class="sidebar-menu"
        router
        background-color="#545c64"
        text-color="#fff"
        active-text-color="#ffd04b"
      >
        <div class="logo">
          EPUB Translator
        </div>
        
        <el-menu-item index="/translation">
          <el-icon><Document /></el-icon>
          <span>翻译服务</span>
        </el-menu-item>
        
        <el-menu-item index="/recharge">
          <el-icon><CreditCard /></el-icon>
          <span>充值中心</span>
        </el-menu-item>
        
        <el-menu-item index="/admin" v-if="isAdmin">
          <el-icon><Setting /></el-icon>
          <span>系统管理</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header class="header">
        <div class="header-content">
          <span>欢迎, {{ authStore.user?.email }}</span>
          <el-button @click="handleLogout" type="danger" plain>
            退出登录
          </el-button>
        </div>
      </el-header>
      
      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Document, CreditCard, Setting } from '@element-plus/icons-vue'
import { useAuthStore } from '../stores/auth.js'

const router = useRouter()
const authStore = useAuthStore()

const isAdmin = ref(false)

const handleLogout = async () => {
  try {
    await authStore.signOut()
    ElMessage.success('退出成功')
    router.push('/login')
  } catch (error) {
    ElMessage.error('退出失败')
  }
}

onMounted(() => {
  if (!authStore.user) {
    router.push('/login')
  }
})
</script>

<style scoped>
.dashboard-container {
  height: 100vh;
}

.sidebar-menu {
  height: 100vh;
  border-right: none;
}

.logo {
  padding: 20px;
  text-align: center;
  color: #fff;
  font-weight: bold;
  font-size: 16px;
  border-bottom: 1px solid #434a50;
  margin-bottom: 10px;
}

.header {
  background-color: #fff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
  padding: 0 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.main-content {
  background-color: #f5f5f5;
  padding: 20px;
}
</style>