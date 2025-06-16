import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth.js'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/LoginView.vue')
  },
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('../views/DashboardView.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        redirect: '/translation'
      },
      {
        path: '/translation',
        name: 'Translation',
        component: () => import('../views/TranslationView.vue')
      },
      {
        path: '/recharge',
        name: 'Recharge',
        component: () => import('../views/RechargeView.vue')
      },
      {
        path: '/admin',
        name: 'Admin',
        component: () => import('../views/AdminView.vue'),
        meta: { requiresAdmin: true }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  
  if (to.matched.some(record => record.meta.requiresAuth)) {
    if (!authStore.user) {
      next('/login')
      return
    }
  }
  
  if (to.matched.some(record => record.meta.requiresAdmin)) {
    // Admin check would need to be implemented based on user profile
    // For now, just check if user exists
    if (!authStore.user) {
      next('/login')
      return
    }
  }
  
  next()
})

export default router