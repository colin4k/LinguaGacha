### **项目背景与目标**

您是一位经验丰富的全栈开发工程师。当前，我们有一个核心的 Python 脚本 api.py，它封装了调用大语言模型进行 EPUB 文件翻译和**文件存储**的核心逻辑。您的任务是基于此脚本，并以 **Supabase** 作为数据库和认证核心，构建一个功能完善、生产可用的 Web 应用。

### **一、核心技术栈**

* **后端框架**: FastAPI (利用其高性能和 Pydantic 集成)。  
* **前端框架**: Vue 3 (使用 Composition API 和 \<script setup\>)。  
* **UI 库**: Element Plus。  
* **数据库与认证服务**: **Supabase** (提供 PostgreSQL 数据库、用户认证、实时数据订阅)。  
* **文件存储**: **服务器本地文件系统**，通过后端 API 进行管理。  
* **异步任务队列**: **内置的、基于数据库持久化的内存队列**，以降低资源消耗。

### **二、项目文件结构**

/epub-translator-app  
├── /backend  
│   ├── /app  
│   │   ├── /api          \# API 路由  
│   │   ├── /core         \# 核心配置 (含 Supabase 客户端、队列初始化)  
│   │   ├── /schemas      \# Pydantic 数据校验模型  
│   │   ├── /services     \# 业务逻辑服务 (如翻译、支付、后台任务处理)  
│   │   ├── \_\_init\_\_.py  
│   │   └── main.py       \# FastAPI 应用入口  
│   ├── /uploads        \# 新增：用于存储上传文件的目录  
│   ├── requirements.txt  
│   └── .env            \# 环境变量文件  
│  
└── /frontend  
    ├── /src  
    │   ├── /api          \# 后端 API 请求封装 (Axios)  
    │   ├── /assets       \# 静态资源  
    │   ├── /components   \# 可复用组件  
    │   ├── /router       \# 路由配置  
    │   ├── /stores       \# Pinia 状态管理  
    │   ├── /supabase.js  \# Supabase 客户端初始化  
    │   ├── /views        \# 页面级组件  
    │   ├── App.vue  
    │   └── main.js  
    ├── package.json  
    └── vite.config.js

### **三、后端详细需求 (Python/FastAPI)**

1. **核心配置**:  
   * 在 .env 文件中配置 SUPABASE\_URL, SUPABASE\_SERVICE\_KEY 和 TRANSLATION\_WORKERS。  
2. **API 接口定义**:  
   * **认证**: 后端创建一个可依赖的函数，用于验证前端请求头中由 Supabase 签发的 JWT，并从中解析出用户 ID。  
   * **支付与点数**:  
     * POST /api/recharge: 接收充值金额，创建本地订单记录（状态为 pending），调用第三方支付接口生成支付二维码信息并返回给前端。  
     * GET /api/recharge/status?order\_id={order\_id}: 前端轮询此接口，用于查询特定订单在数据库中的支付状态。  
     * POST /api/recharge/notify: 接收第三方支付平台的异步回调。在验证回调合法性后，更新订单状态为 completed，并使用 Supabase 客户端为用户增加相应的点数。  
   * **翻译任务**:  
     * POST /api/translations/upload:  
       * 需验证 Supabase JWT。  
       * 接收文件 (FastAPI.UploadFile)，校验用户点数。  
       * **关键变更**: 调用 api.py 提供的函数将文件保存到服务器本地的指定目录 (例如 /uploads/{user\_id}/)。  
       * 在 translation\_tasks 表中创建一条状态为 pending 的记录，其中 storage\_path 字段保存文件的**本地服务器路径**。  
       * 将新任务的 task\_id 放入一个全局的 asyncio.Queue 中。  
     * GET /api/translations: 获取当前用户的所有翻译任务列表。  
     * GET /api/translations/{task\_id}/download:  
       * **关键变更**: 从数据库获取任务的**本地** translated\_storage\_path，从服务器文件系统读取文件并以文件流形式安全返回。  
   * **管理员配置**:  
     * GET /api/admin/llm-settings: 需要管理员权限。从 llm\_config 表中查询并返回当前的 base\_url, api\_key, model。  
     * PUT /api/admin/llm-settings: 需要管理员权限。接收新的配置信息并更新到 llm\_config 表中。  
3. **数据库表 (Supabase Tables)**:  
   * 表结构保持不变，但请注意 storage\_path 和 translated\_storage\_path 现在存储的是**服务器上的相对或绝对文件路径**。  
   * profiles, translation\_tasks, recharge\_orders, llm\_config。  
   * 为 translation\_tasks 表开启 **Realtime** 功能。  
4. **内置持久化队列实现**:  
   * **工作流程**:  
     1. 从队列获取 task\_id。  
     2. 使用 Supabase 客户端将数据库中该任务的状态更新为 processing。  
     3. **关键变更**: 从 storage\_path 字段获取**本地文件路径**并读取文件。  
     4. 调用核心翻译逻辑。  
     5. **成功**: 将翻译好的文件保存到本地，并将**新的本地路径**更新到 translated\_storage\_path 字段，然后将任务状态更新为 completed。  
     6. **失败**: 将任务状态更新为 failed。  
   * **持久化与恢复**: 应用重启时，查询数据库中所有 pending 或 processing 状态的任务，并将其 task\_id 重新加入内存队列。

### **四、前端详细需求 (Vue 3\)**

1. **Supabase 集成**:  
   * 在 src/supabase.js 中，使用 createClient 方法初始化 Supabase 客户端，并从环境变量 (.env.local) 中读取 SUPABASE\_URL 和 SUPABASE\_ANON\_KEY。  
2. **页面/视图 (Views)**:  
   * LoginView.vue: 提供电子邮件和密码输入框。分别调用 Supabase JS 库的 signInWithPassword() 和 signUp() 方法处理登录和注册逻辑。  
   * DashboardView.vue: 作为应用主布局，包含侧边栏导航和 \<router-view\>。在组件挂载时，监听 Supabase 的 onAuthStateChange 事件，若用户未登录，则自动跳转到登录页。  
   * TranslationView.vue:  
     * **关键变更**: 使用 Axios 将文件以 multipart/form-data 格式提交到后端的 /api/translations/upload 接口。  
     * **实时任务列表**: 使用 supabase.channel(...).on(...).subscribe() 订阅 translation\_tasks 表的变化，实时自动更新任务列表。  
     * 提供下载按钮，请求后端的安全下载接口。  
   * RechargeView.vue: 提供金额输入框，点击按钮后调用 /api/recharge 接口，在模态框中显示返回的支付二维码，并启动定时器轮询支付状态。  
   * AdminView.vue: 提供一个表单，用于显示和修改大模型配置。页面加载时获取配置，修改后提交保存。此页面需要路由守卫保护，仅限管理员访问。  
3. **核心功能实现**:  
   * **状态管理 (Pinia)**: 创建一个 authStore，用于存储从 Supabase 获取的用户信息和会话。在 onAuthStateChange 事件回调中更新此 Store。  
   * **API 请求 (Axios)**: 创建一个 Axios 实例，并配置请求拦截器。在拦截器中，从 Supabase 获取当前会话的 access\_token，并将其附加到每个发往后端 API 的请求头中 (Authorization: Bearer {token})。