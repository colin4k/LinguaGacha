import { defineStore } from 'pinia'
import { ref } from 'vue'
import { supabase } from '../supabase.js'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const session = ref(null)
  const loading = ref(true)

  const setAuth = (newSession) => {
    session.value = newSession
    user.value = newSession?.user || null
    loading.value = false
  }

  const signIn = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    
    if (error) throw error
    return data
  }

  const signUp = async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    
    if (error) throw error
    return data
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    
    user.value = null
    session.value = null
  }

  const initialize = () => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setAuth(session)
    })

    supabase.auth.onAuthStateChange((_event, session) => {
      setAuth(session)
    })
  }

  return {
    user,
    session,
    loading,
    signIn,
    signUp,
    signOut,
    initialize
  }
})