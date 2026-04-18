---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
    font-size: 26px;
    padding: 34px 46px 58px 46px;
    line-height: 1.35;
    color: #18324a;
    background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
  }
  section.lead {
    background: radial-gradient(circle at top right, #dcecff 0%, #eef5ff 24%, #ffffff 62%);
    color: #14324b;
  }
  h1 {
    font-size: 38px;
    color: #103b66;
    margin: 0 0 10px 0;
    line-height: 1.2;
  }
  h2 {
    font-size: 28px;
    color: #0f4c81;
    margin: 0 0 8px 0;
  }
  h3 {
    font-size: 22px;
    color: #1d5d91;
    margin: 0 0 6px 0;
  }
  p, li { font-size: 23px; }
  ul { margin: 6px 0 0 18px; }
  li { margin: 6px 0; }
  strong { color: #0b558d; }
  .subtitle {
    font-size: 20px;
    color: #5b7083;
    margin-top: 4px;
  }
  .small {
    font-size: 16px;
    color: #6a7885;
  }
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 12px;
  }
  .grid3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-top: 12px;
  }
  .card {
    background: #ffffff;
    border: 1.5px solid #d7e6f3;
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(16,59,102,0.06);
  }
  .accent {
    background: linear-gradient(180deg, #edf6ff 0%, #ffffff 100%);
    border: 1.5px solid #b8d8f0;
  }
  .warn {
    background: linear-gradient(180deg, #fff6eb 0%, #ffffff 100%);
    border: 1.5px solid #f3d2a8;
  }
  .green {
    background: linear-gradient(180deg, #eefbf5 0%, #ffffff 100%);
    border: 1.5px solid #b8e2c8;
  }
  .hero {
    margin-top: 18px;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 20px;
    align-items: center;
  }
  .hero-box {
    background: rgba(255,255,255,0.86);
    border: 1.5px solid #d4e4f4;
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(16,59,102,0.08);
  }
  .metric {
    text-align: center;
    padding: 12px 10px;
    border-radius: 18px;
    background: #ffffff;
    border: 1.5px solid #d7e6f3;
  }
  .metric b {
    display: block;
    font-size: 28px;
    color: #103b66;
    margin-bottom: 4px;
  }
  .flow {
    margin-top: 14px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    align-items: stretch;
  }
  .step {
    background: #ffffff;
    border: 1.5px solid #d7e6f3;
    border-radius: 18px;
    padding: 14px 16px;
    min-height: 128px;
    box-shadow: 0 8px 18px rgba(16,59,102,0.06);
  }
  .pilltitle {
    display: inline-block;
    background: #103b66;
    color: white;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 16px;
    margin-bottom: 8px;
  }
  .highlight {
    margin-top: 12px;
    padding: 14px 18px;
    border-left: 6px solid #2a7abf;
    background: linear-gradient(90deg, #edf6ff 0%, #ffffff 100%);
    border-radius: 0 16px 16px 0;
    font-size: 22px;
  }
  .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 10px;
  }
  .cite {
    position: absolute;
    left: 46px;
    right: 46px;
    bottom: 12px;
    font-size: 13px;
    color: #708292;
    line-height: 1.2;
  }
  .footer-note {
    position: absolute;
    right: 46px;
    top: 26px;
    font-size: 14px;
    color: #6f8292;
    background: rgba(255,255,255,0.75);
    border: 1px solid #d9e6f2;
    border-radius: 999px;
    padding: 4px 10px;
  }
  .img-wide {
    margin-top: 12px;
    display: flex;
    justify-content: center;
  }
  .img-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1.5px solid #d7e6f3;
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(16,59,102,0.08);
    text-align: center;
  }
  .img-card img {
    width: 100%;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(16,59,102,0.10);
  }
  .img-wide .img-card {
    width: 54%;
    max-width: 980px;
  }
  .paper-title {
    margin-top: 10px;
    padding: 10px 16px;
    background: linear-gradient(90deg, #edf6ff 0%, #ffffff 100%);
    border: 1.5px solid #cfe0f3;
    border-radius: 16px;
    font-size: 18px;
    line-height: 1.35;
    color: #20486d;
  }
  .paper-label {
    display: inline-block;
    margin-right: 10px;
    padding: 3px 10px;
    border-radius: 999px;
    background: #103b66;
    color: #fff;
    font-size: 14px;
    vertical-align: middle;
  }
  .lit-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 14px;
    align-items: stretch;
  }
  .lit-card { min-height: 255px; }
  .lit-card li {
    margin: 8px 0;
    font-size: 21px;
    line-height: 1.4;
  }
  .note {
    margin-top: 10px;
    font-size: 18px;
    color: #666;
    text-align: center;
  }
---
<!-- _class: lead -->

<div class="footer-note">组会汇报 / 研究设想</div>

# 面向真实路网的电动汽车充电协同调度研究
## 从文献梳理到研究方向凝练

<div class="subtitle">关键词：真实路网｜协同调度｜闭环优化</div>

<div class="hero">
  <div class="hero-box">
    <div class="pilltitle">一句话概括</div>
    <div style="font-size:26px; line-height:1.45;">
      在真实城市路网下，研究<strong>车辆选站、站点排队与局部电网负荷</strong>之间的耦合关系，
      重点关注<strong>更合理的动态选站与协同优化</strong>。
    </div>
  </div>
  <div>
    <div class="grid2">
      <div class="metric"><b>车</b>位置 / SOC / 出行</div>
      <div class="metric"><b>路</b>拓扑 / 拥堵 / 时间</div>
      <div class="metric"><b>站</b>排队 / 价格 / 容量</div>
      <div class="metric"><b>网</b>负荷 / 电压 / 风险</div>
    </div>
  </div>
</div>

<div class="small" style="margin-top:18px;">汇报人：吴家鸿　导师：卢少锋</div>

---
<div class="footer-note">选题灵感</div>

<div class="img-wide">
  <div class="img-card">
    <img src="image/组会汇报_车网协同调度_最终版_美化_含建模与指标.marp/1774442807695.png" />
    <div class="note">图：选题灵感</div>
  </div>
</div>

---
<div class="footer-note">研究问题</div>

# 1. 我想解决的核心问题是什么？

<div class="grid2">
  <div class="card accent">
    <h3>现实中的充电决策</h3>
    <ul>
      <li>不是简单“找最近的站”</li>
      <li>会同时影响<strong>出行效率、站点排队、电网负荷</strong></li>
      <li>单车最优选择，可能导致系统整体失衡</li>
    </ul>
  </div>
  <div class="card warn">
    <h3>本课题关注的问题</h3>
    <ul>
      <li>在<strong>真实路网</strong>条件下动态分配充电站</li>
      <li>兼顾<strong>用户体验、站点运行、电网安全</strong></li>
      <li>避免局部拥堵、排队和负荷聚集</li>
    </ul>
  </div>
</div>

<div class="highlight">
本质上，这是一个<strong>车 - 路 - 站 - 网耦合的动态协同调度问题</strong>，而不是单一站点推荐问题。
</div>

<div class="cite">[引用] Sun et al., 2019; Cui et al., 2021; Liu et al., 2022.</div>



---
<div class="footer-note">文献认识</div>

# 2. 现有研究已经做到什么？还缺什么？

<div class="paper-title"><span class="paper-label">文献 1</span>Battery Health-Informed and Policy-Aware Deep Reinforcement Learning for EV-Facilitated Distribution Grid Optimal Policy</div>

<div class="lit-layout">
  <div class="card green lit-card">
    <h3>目前进展</h3>
    <ul>
      <li>已能用深度强化学习实现 EV 充放电调度。</li>
      <li>已将 EV 纳入配电网联合优化框架。</li>
      <li>开始从“只看充电成本”转向“兼顾电网运行”。</li>
      <li>验证了在线调度能力和一定可扩展性。</li>
    </ul>
  </div>
  <div class="card warn lit-card">
    <h3>目前不足</h3>
    <ul>
      <li>仍偏集中式，难兼顾用户自主性与隐私。</li>
      <li>个体 EV 对系统策略的真实响应难保证。</li>
      <li>大规模场景下实时决策压力较大。</li>
      <li>对真实道路交通等因素考虑仍有限。</li>
    </ul>
  </div>
</div>

<div class="cite">[引用] Xie et al., 2025.</div>

---
<div class="footer-note">文献认识</div>

# 2. 现有研究已经做到什么？还缺什么？

<div class="paper-title"><span class="paper-label">文献 2</span>Shortest-Path-Based Deep Reinforcement Learning for EV Charging Routing Under Stochastic Traffic Condition and Electricity Prices</div>

<div class="lit-layout">
  <div class="card green lit-card">
    <h3>目前进展</h3>
    <ul>
      <li>已把充电路径问题建成随机环境下的在线决策。</li>
      <li>实现了“走哪条路 + 去哪个站充电”的统一建模。</li>
      <li>可在交通状态、电价波动等条件下动态决策。</li>
      <li>体现了路径与补能过程的耦合。</li>
    </ul>
  </div>
  <div class="card warn lit-card">
    <h3>目前不足</h3>
    <ul>
      <li>更偏单车路径选择，对电网侧协同较少。</li>
      <li>对电池健康和长期退化未深入建模。</li>
      <li>大规模场景下仍面临状态空间压力。</li>
      <li>对多车竞争和系统级负荷均衡讨论不足。</li>
    </ul>
  </div>
</div>

<div class="cite">[引用] Jin & Xu, 2022.</div>

---
<div class="footer-note">文献认识</div>

# 2. 现有研究已经做到什么？还缺什么？

<div class="paper-title"><span class="paper-label">文献 3</span>Deep Reinforcement Learning for EV Charging Navigation by Coordinating Smart Grid and Intelligent Transportation System</div>

<div class="lit-layout">
  <div class="card green lit-card">
    <h3>目前进展</h3>
    <ul>
      <li>将 EV 充电导航建模为随机环境下的 MDP。</li>
      <li>结合路况、价格和等待时间做在线选站与路径决策。</li>
      <li>不依赖不确定性先验，可适应随机变化。</li>
      <li>在实际城市原型场景中验证了有效性。</li>
    </ul>
  </div>
  <div class="card warn lit-card">
    <h3>目前不足</h3>
    <ul>
      <li>主要从单车视角出发，未刻画多车竞争。</li>
      <li>场景规模较小，可扩展性仍需进一步证明。</li>
      <li>对配电网运行约束与电池老化考虑不足。</li>
      <li>随机环境建模仍偏理想化。</li>
    </ul>
  </div>
</div>

<div class="cite">[引用] Qian et al., 2020.</div>

---
<div class="footer-note">归纳总结</div>

# 3. 从文献里，我归纳出的两个关键不足

<div class="grid2">
  <div class="card warn">
    <h3>不足 1：缺少闭环</h3>
    <ul>
      <li>有的工作偏交通侧，只解决“去哪充、怎么走”</li>
      <li>有的工作偏电网侧，只关注系统层联合优化</li>
      <li>真正把<strong>交通、站点、电网</strong>贯通起来的闭环框架还不够多</li>
    </ul>
  </div>
  <div class="card warn">
    <h3>不足 2：缺少系统协调</h3>
    <ul>
      <li>不少研究仍偏单车或单侧优化</li>
      <li>对多车竞争导致的<strong>排队、拥堵、负荷聚集</strong>考虑不足</li>
      <li>难以回答“系统整体会不会更好”这个问题</li>
    </ul>
  </div>
</div>

<div class="highlight">
所以我后续更想抓住两点：<strong>闭环优化</strong>，以及<strong>系统协调</strong>。
</div>

---
<div class="footer-note">研究方向</div>

# 4. 我目前初步选定的研究方向

<div class="card accent" style="margin-top:18px; padding:22px 24px;">
  <div style="font-size:28px; line-height:1.5;">
    面向<strong>真实路网</strong>的电动汽车充电协同调度：
    在交通、站点和电网约束共同作用下，研究更合理的<strong>动态选站与协同优化</strong>。
  </div>
</div>

<div class="flow">
  <div class="step"><div class="pilltitle">重点 1</div><strong>真实路网</strong><br>让路径可达关系、拥堵传播和出行代价更贴近实际</div>
  <div class="step"><div class="pilltitle">重点 2</div><strong>协同调度</strong><br>不只看单车最优，还关注站点排队和局部电网负荷</div>
  <div class="step"><div class="pilltitle">重点 3</div><strong>闭环优化</strong><br>把选站决策与站点执行、电网反馈衔接起来</div>
</div>

<div class="highlight">
这次组会先把主线收敛到<strong>真实路网 + 协同调度 + 闭环优化</strong>，后续再考虑其他扩展问题。
</div>

---
<div class="footer-note">总体思路</div>

# 5. 目前拟采用的总体思路

<div class="flow">
  <div class="step"><div class="pilltitle">输入</div>车辆位置 / SOC<br>路网状态<br>站点排队 / 价格<br>电网负荷反馈</div>
  <div class="step"><div class="pilltitle">决策</div>结合图结构表示空间关联<br>输出目标站点或候选站点排序</div>
  <div class="step"><div class="pilltitle">反馈</div>根据排队、负荷、电压等指标<br>形成下一轮调度修正</div>
</div>

<div class="grid2" style="margin-top:18px;">
  <div class="card accent">
    <h3>想解决的问题</h3>
    “推荐哪个站”往往不够，关键在于<strong>能否在系统层面更合理</strong>。
  </div>
  <div class="card green">
    <h3>希望达到的效果</h3>
    同时改善<strong>用户体验、站点拥堵和局部电网运行</strong>。
  </div>
</div>

---
<div class="footer-note">研究计划</div>

# 6. 接下来准备怎么推进

<div class="grid3">
  <div class="card accent">
    <h3>阶段一</h3>
    <ul>
      <li>把真实路网场景搭起来</li>
      <li>统一车辆、站点、电网状态描述</li>
      <li>先形成可运行的基础环境</li>
    </ul>
  </div>
  <div class="card green">
    <h3>阶段二</h3>
    <ul>
      <li>实现动态选站与协同调度方法</li>
      <li>与现有单车/单侧方法做对比</li>
      <li>验证是否能缓解排队与负荷聚集</li>
    </ul>
  </div>
  <div class="card warn">
    <h3>阶段三</h3>
    <ul>
      <li>在主线跑通后再考虑进一步扩展</li>
      <li>例如羊群效应抑制、隐私保护等</li>
      <li>避免前期研究主线过于发散</li>
    </ul>
  </div>
</div>

---

# 7. 目前我是怎么建模的？

<div class="grid2">
  <div class="card accent">
    <h3>系统架构：车 - 路 - 站 - 网 - 边</h3>
    <ul>
      <li><strong>交通网</strong>：真实路网图，路段时间随流量动态变化</li>
      <li><strong>配电网</strong>：辐射型配电网，站点负荷映射到母线节点</li>
      <li><strong>充电站</strong>：交通服务节点 + 电力负荷节点双重角色</li>
    </ul>
  </div>
  <div class="card green">
    <h3>交通侧成本建模</h3>
    <ul>
      <li><strong>广义成本</strong> = 行驶时间 + 站内服务时间 + 充电费用</li>
      <li><strong>行驶时间</strong>：采用 BPR 函数描述拥堵下的时变通行时间</li>
      <li><strong>服务时间</strong>：排队等待 + 实际充电时间</li>
      <li><strong>充电费用</strong>：由实时电价和补能需求共同决定</li>
    </ul>
  </div>
</div>

---
<div class="grid2" style="margin-top:14px;">
  <div class="card warn">
    <h3>电网侧建模</h3>
    <ul>
      <li>采用 <strong>DistFlow</strong> 描述辐射型配电网潮流</li>
      <li>跟踪节点负荷、电压偏移、线路损耗</li>
      <li>实时电价采用 <strong>TOU 均值 + 随机波动</strong> 的方式生成</li>
      <li>电网状态既是约束，也是对交通决策的反馈信号</li>
    </ul>
  </div>
  <div class="card">
    <h3>联合优化目标</h3>
    <ul>
      <li>用户总出行与充电成本</li>
      <li>排队等待代价</li>
      <li>配电网运行成本</li>
      <li>负荷波动 / 峰谷差惩罚</li>
      <li>通信代价（作为隐私保护扩展项可选加入）</li>
    </ul>
  </div>
</div>

<div class="highlight">
当前建模不是只做“站点推荐”，而是把<strong>交通出行、站内服务、电网反馈和后续分布式扩展</strong>放进同一个协同框架里。
</div>

<div class="cite">[建模依据] 车-路-站-网-边耦合架构、交通侧一体化成本、DistFlow 电网模型与联合优化目标，来自你上传的模型章初稿。</div>

---

<div class="footer-note">选题灵感</div>
<div class="img-wide">
  <div class="img-card">
    <img src="image/组会汇报_车网协同调度_最终版_美化_含建模与指标.marp/组会.png" />
    <div class="note">图：广州珠江新城真实地图</div>
  </div>
</div>



---
<div class="footer-note">总结</div>

# 8. 总结

<div class="grid2">
  <div class="card accent">
    <h3>现阶段我想讲清楚的</h3>
    <ul>
      <li>这个问题不是简单的充电站推荐</li>
      <li>而是车、路、站、网耦合下的协同调度问题</li>
      <li>现有研究已有基础，但仍缺闭环和系统协调</li>
    </ul>
  </div>
  <div class="card green">
    <h3>当前收敛后的主线</h3>
    <ul>
      <li><strong>真实路网</strong></li>
      <li><strong>协同调度</strong></li>
      <li><strong>闭环优化</strong></li>
    </ul>
  </div>
</div>

<div class="highlight">
我后续会先把这条主线做扎实，再在此基础上考虑羊群效应和隐私保护等扩展方向。
</div>
