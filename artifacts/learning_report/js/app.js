/**
 * BEV / OCC Perception Textbook - Master Router & Application Controller (Audited & Authentic)
 */

let CURRICULUM_DATA = null;
let currentMode = 'full';
let currentSubpageId = 'home';
let masterySet = new Set();

// Active 3D instances
let lidarViewerInstance = null;
let occViewerInstance = null;
let sensorRigInstance = null;

async function initApp() {
  try {
    const res = await fetch('learning_report/data/curriculum.json');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    CURRICULUM_DATA = await res.json();
  } catch (e) {
    console.warn('[App] Using inline fallback curriculum data:', e);
    CURRICULUM_DATA = getFallbackCurriculum();
  }

  // Load mastery state from localStorage
  try {
    const saved = localStorage.getItem('bev_occ_mastery');
    if (saved) {
      masterySet = new Set(JSON.parse(saved));
    }
  } catch (e) {}

  buildSidebar();
  setupEventListeners();

  // Listen to hash change for routing
  window.addEventListener('hashchange', handleHashChange);
  
  // Initial route
  const hash = window.location.hash.replace('#', '');
  if (hash && document.getElementById(`subpage-${hash}`)) {
    navigateTo(hash);
  } else if (document.getElementById('subpage-home')) {
    navigateTo('home');
  } else {
    navigateTo('ch01-sub01');
  }
}

function buildSidebar() {
  if (!CURRICULUM_DATA) return;
  const container = document.getElementById('sidebar-tree-container');
  if (!container) return;

  container.innerHTML = '';

  // Top Home Navigation Button
  const homeItem = document.createElement('div');
  homeItem.className = 'sidebar-home-item';
  homeItem.innerHTML = `
    <a href="#home" class="subpage-link ${currentSubpageId === 'home' ? 'active' : ''}" id="link-home" style="display: flex; align-items: center; gap: 0.6rem; padding: 0.65rem 0.9rem; font-weight: 600; color: var(--accent-cyan); background: rgba(0, 229, 255, 0.08); border-radius: 6px; margin-bottom: 0.75rem; border: 1px solid rgba(0, 229, 255, 0.2);">
      <span>🏠 课程学习地图 (Home)</span>
    </a>
  `;
  container.appendChild(homeItem);

  const currentModeChapters = CURRICULUM_DATA.chapters;

  currentModeChapters.forEach(ch => {
    // Filter subpages based on mode
    const visibleSubpages = ch.subpages.filter(sp => {
      if (currentMode === 'fast') return sp.mode.includes('fast');
      if (currentMode === 'interview') return sp.mode.includes('interview');
      return true; // full mode includes all
    });

    if (visibleSubpages.length === 0) return;

    const group = document.createElement('div');
    group.className = 'chapter-group open';
    group.id = `group-${ch.id}`;

    const header = document.createElement('div');
    header.className = 'chapter-header';
    header.innerHTML = `
      <div class="chapter-title-wrap">
        <span class="chapter-badge">${ch.number}</span>
        <span>${ch.title}</span>
      </div>
      <span class="chapter-arrow">▼</span>
    `;

    header.addEventListener('click', () => {
      group.classList.toggle('open');
      const arrow = header.querySelector('.chapter-arrow');
      if (arrow) arrow.innerText = group.classList.contains('open') ? '▼' : '▶';
    });

    const list = document.createElement('ul');
    list.className = 'subpage-list';

    visibleSubpages.forEach(sp => {
      const isCompleted = masterySet.has(sp.id);
      const li = document.createElement('li');
      li.className = 'subpage-item';
      li.innerHTML = `
        <a href="#${sp.id}" class="subpage-link ${sp.id === currentSubpageId ? 'active' : ''}" id="link-${sp.id}">
          <span>${sp.title}</span>
          <span class="check-icon ${isCompleted ? 'completed' : ''}" id="check-${sp.id}">✓</span>
        </a>
      `;
      list.appendChild(li);
    });

    group.appendChild(header);
    group.appendChild(list);
    container.appendChild(group);
  });

  updateProgress();
}

function navigateTo(subpageId) {
  currentSubpageId = subpageId;
  window.location.hash = subpageId;

  // Hide all subpages
  document.querySelectorAll('.subpage-content-section').forEach(el => {
    el.style.display = 'none';
  });

  // Show target subpage
  const targetEl = document.getElementById(`subpage-${subpageId}`);
  if (targetEl) {
    targetEl.style.display = 'block';
  }

  // Update sidebar active link and open its group
  document.querySelectorAll('.subpage-link').forEach(link => link.classList.remove('active'));
  const activeLink = document.getElementById(`link-${subpageId}`);
  if (activeLink) {
    activeLink.classList.add('active');
    const group = activeLink.closest('.chapter-group');
    if (group) group.classList.add('open');
  }

  // Update Breadcrumb
  updateBreadcrumb(subpageId);

  // Update Bottom Bar Mastery Checkbox
  const masteryCheck = document.getElementById('bottom-mastery-checkbox');
  if (masteryCheck) {
    masteryCheck.checked = masterySet.has(subpageId);
  }

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Trigger KaTeX dynamic math rendering on target subpage
  setTimeout(() => {
    if (targetEl) renderMath(targetEl);
  }, 40);

  // Post-navigation view hooks
  handleSubpageHooks(subpageId);
}

function renderMath(targetEl) {
  const el = targetEl || document.body;
  if (typeof renderMathInElement === 'function') {
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false,
        errorColor: '#FF5252'
      });
    } catch (e) {
      console.warn('[KaTeX] renderMath error:', e);
    }
  }
}
window.renderMath = renderMath;

function handleHashChange() {
  const hash = window.location.hash.replace('#', '');
  if (hash && hash !== currentSubpageId) {
    navigateTo(hash);
  }
}

function handleSubpageHooks(subpageId) {
  setTimeout(() => {
    // 1. LiDAR 3D Viewer
    if (subpageId === 'ch01-sub03') {
      if (!lidarViewerInstance && document.getElementById('lidar-canvas-3d')) {
        lidarViewerInstance = new LidarViewer3D('lidar-canvas-3d', 'lidar-stats', 'lidar-metadata-banner');
      } else if (lidarViewerInstance) {
        lidarViewerInstance.render();
      }
    } 
    // 2. Occupancy 3D Viewer
    else if (subpageId === 'ch09-sub02' || subpageId === 'ch10-sub03' || subpageId === 'ch11-sub04') {
      if (!occViewerInstance && document.getElementById('occ-canvas-3d')) {
        occViewerInstance = new OccupancyViewer3D('occ-canvas-3d', 'occ-stats', 'occ-metadata-banner');
      } else if (occViewerInstance) {
        occViewerInstance.render();
      }
    } 
    // 3. Sensor Rig Viewer
    else if (subpageId === 'ch01-sub02' || subpageId === 'ch02-sub01') {
      if (!sensorRigInstance && document.getElementById('rig-canvas-3d')) {
        sensorRigInstance = new SensorRigViewer('rig-canvas-3d', 'rig-info-card');
      } else if (sensorRigInstance) {
        sensorRigInstance.render();
      }
    } 
    // 4. Coordinate Simulator
    else if (subpageId === 'ch02-sub03') {
      if (typeof updateCoordSimulator === 'function') updateCoordSimulator();
    }
    // 5. LSS Explorer
    else if (subpageId === 'ch04-sub02' || subpageId === 'ch05-sub04') {
      if (typeof updateLSSDepthSlider === 'function') updateLSSDepthSlider(27);
    }
    // 6. Temporal Simulator
    else if (subpageId === 'ch06-sub02') {
      if (typeof updateTemporalMode === 'function') updateTemporalMode('aligned');
    }
    // 7. Sparse4D Layer Slider
    else if (subpageId === 'ch08-sub04') {
      if (typeof updateSparse4DLayer === 'function') updateSparse4DLayer(0);
    }
  }, 100);
}

function updateBreadcrumb(subpageId) {
  const breadcrumbEl = document.getElementById('breadcrumb-text');
  if (!breadcrumbEl) return;

  if (subpageId === 'home') {
    breadcrumbEl.innerHTML = `<span style="color: var(--accent-cyan);">🏠 课程学习地图 (Home)</span>`;
    return;
  }

  if (!CURRICULUM_DATA) return;
  for (const ch of CURRICULUM_DATA.chapters) {
    for (const sp of ch.subpages) {
      if (sp.id === subpageId) {
        breadcrumbEl.innerHTML = `<span>${ch.number} ${ch.title}</span> <span class="breadcrumb-separator">/</span> <span style="color: var(--accent-cyan);">${sp.title}</span>`;
        return;
      }
    }
  }
}

function toggleMasteryCurrent() {
  const checkbox = document.getElementById('bottom-mastery-checkbox');
  if (!checkbox) return;

  if (checkbox.checked) {
    masterySet.add(currentSubpageId);
  } else {
    masterySet.delete(currentSubpageId);
  }

  try {
    localStorage.setItem('bev_occ_mastery', JSON.stringify(Array.from(masterySet)));
  } catch (e) {}

  const checkIcon = document.getElementById(`check-${currentSubpageId}`);
  if (checkIcon) {
    checkIcon.classList.toggle('completed', checkbox.checked);
  }

  updateProgress();
}

function updateProgress() {
  if (!CURRICULUM_DATA) return;
  let allCount = 0;
  CURRICULUM_DATA.chapters.forEach(ch => {
    allCount += ch.subpages.length;
  });

  const completedCount = masterySet.size;
  const pct = allCount > 0 ? Math.round((completedCount / allCount) * 100) : 0;

  const bar = document.getElementById('progress-bar-fill');
  const txt = document.getElementById('progress-percent-txt');
  if (bar) bar.style.width = `${pct}%`;
  if (txt) txt.innerText = `${pct}%`;
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  buildSidebar();
  
  if (mode === 'interview') {
    navigateTo('ch12-sub04');
  } else if (mode === 'fast') {
    navigateTo('ch01-sub01');
  }
}

function navigateStep(direction) {
  if (!CURRICULUM_DATA) return;
  if (currentSubpageId === 'home') {
    navigateTo('ch01-sub01');
    return;
  }

  const allSubpages = [];
  CURRICULUM_DATA.chapters.forEach(ch => {
    ch.subpages.forEach(sp => {
      if (currentMode === 'fast' && !sp.mode.includes('fast')) return;
      if (currentMode === 'interview' && !sp.mode.includes('interview')) return;
      allSubpages.push(sp.id);
    });
  });

  const idx = allSubpages.indexOf(currentSubpageId);
  if (idx === -1) return;

  const newIdx = idx + direction;
  if (newIdx >= 0 && newIdx < allSubpages.length) {
    navigateTo(allSubpages[newIdx]);
  } else if (newIdx < 0) {
    navigateTo('home');
  }
}

function toggleAccordion(id) {
  const item = document.getElementById(id);
  if (!item) return;
  const body = item.querySelector('.accordion-body');
  const arrow = item.querySelector('.accordion-arrow');
  if (!body) return;
  
  const isOpen = body.style.display === 'block';
  body.style.display = isOpen ? 'none' : 'block';
  if (arrow) arrow.innerText = isOpen ? '▶' : '▼';
}

function setupEventListeners() {
  // Mode Buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  // Prev / Next Page Buttons
  const prevBtn = document.getElementById('bottom-prev-btn');
  const nextBtn = document.getElementById('bottom-next-btn');
  if (prevBtn) prevBtn.addEventListener('click', () => navigateStep(-1));
  if (nextBtn) nextBtn.addEventListener('click', () => navigateStep(1));

  // Mastery Checkbox
  const masteryCheck = document.getElementById('bottom-mastery-checkbox');
  if (masteryCheck) masteryCheck.addEventListener('change', toggleMasteryCurrent);

  // Search input
  const searchInput = document.getElementById('site-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q) return;
      if (!CURRICULUM_DATA) return;
      for (const ch of CURRICULUM_DATA.chapters) {
        for (const sp of ch.subpages) {
          if (sp.title.toLowerCase().includes(q)) {
            navigateTo(sp.id);
            return;
          }
        }
      }
    });
  }

  // Delegated click listener for dashed concept terms (.term-link)
  document.addEventListener('click', (e) => {
    const termEl = e.target.closest('.term-link');
    if (termEl) {
      const conceptKey = termEl.dataset.concept;
      if (conceptKey && typeof openConceptModal === 'function') {
        e.preventDefault();
        openConceptModal(conceptKey);
      }
    }
  });
}

// Global Sensor Rig view preset helpers
window.setRigPreset = function(preset) {
  if (sensorRigInstance && typeof sensorRigInstance.resetView === 'function') {
    sensorRigInstance.resetView(preset);
  }
};
window.selectRigSensor = function(channelId) {
  if (sensorRigInstance && typeof sensorRigInstance.selectSensor === 'function') {
    sensorRigInstance.selectSensor(channelId);
  }
};

// Fallback curriculum structure if fetch is blocked in file:// protocol
function getFallbackCurriculum() {
  return {
    chapters: [
      { id: "ch01", number: "01", title: "数据闭环：nuScenes 真实多模态结构", subpages: [
        { id: "ch01-sub01", title: "1.1 一帧自动驾驶数据到底是什么？", mode: ["fast", "full"] },
        { id: "ch01-sub02", title: "1.2 六路环视相机（Surround Cameras）", mode: ["full"] },
        { id: "ch01-sub03", title: "1.3 LiDAR 原始二进制解析与点云特性", mode: ["fast", "full"] },
        { id: "ch01-sub04", title: "1.4 5 路毫米波雷达（Radar）的多普勒测量", mode: ["full"] },
        { id: "ch01-sub05", title: "1.5 Keyframe（关键帧）与 Sweeps（时序子帧）", mode: ["full"] }
      ]},
      { id: "ch02", number: "02", title: "坐标与几何：从 3D 物理世界到 2D 像素", subpages: [
        { id: "ch02-sub01", title: "2.1 自动驾驶四大坐标系（Sensor/Ego/Global/Cam）", mode: ["fast", "full"] },
        { id: "ch02-sub02", title: "2.2 外参刚体变换与内参透视投影推导", mode: ["fast", "full"] },
        { id: "ch02-sub03", title: "2.3 真实 LiDAR 点投影到相机：分步动画模拟器", mode: ["fast", "full"] },
        { id: "ch02-sub04", title: "2.4 3D 边界框（Bounding Box）相机视角重投影", mode: ["full"] }
      ]},
      { id: "ch03", number: "03", title: "BEV 基础：为什么自动驾驶需要鸟瞰图？", subpages: [
        { id: "ch03-sub01", title: "3.1 透视视角的深度缺失与尺度畸变", mode: ["full"] },
        { id: "ch03-sub02", title: "3.2 鸟瞰视角（BEV）的物理优势", mode: ["fast", "full"] },
        { id: "ch03-sub03", title: "3.3 逆透视映射（IPM）vs 深度学习视图变换", mode: ["full"] }
      ]},
      { id: "ch04", number: "04", title: "BEVDet：最简自底向上 2D➔3D 基线", subpages: [
        { id: "ch04-sub01", title: "4.1 BEVDet 整体架构与数据流", mode: ["full"] },
        { id: "ch04-sub02", "title": "4.2 LSS 视锥构造与 Lift-Splat 体素池化", mode: ["full"] },
        { id: "ch04-sub03", "title": "4.3 真实模型推理输出与 CenterHead 检测", mode: ["full"] },
        { id: "ch04-sub04", "title": "4.4 我们自己的训练：Loss 收敛与 Mini Eval 评测", mode: ["full"] }
      ]},
      { id: "ch05", number: "05", title: "BEVDepth：重点突破显式深度监督", subpages: [
        { id: "ch05-sub01", title: "5.1 为什么无监督深度估计导致 3D 定位崩溃？", mode: ["fast", "full"] },
        { id: "ch05-sub02", title: "5.2 LiDAR 真实深度真值（Depth GT）生成算法", mode: ["fast", "full"] },
        { id: "ch05-sub03", title: "5.3 7 阶段特征流全生命周期拆解", mode: ["fast", "full"] },
        { id: "ch05-sub04", title: "5.4 DepthNet 离散概率分布与 Voxel Pooling v2", mode: ["fast", "full"] },
        { id: "ch05-sub05", title: "5.5 真实子集训练（TRAIN-SUBSET）与 Loss 下降", mode: ["fast", "full"] }
      ]},
      { id: "ch06", number: "06", title: "时序融合：BEVDet4D 与 BEVStereo", subpages: [
        { id: "ch06-sub01", title: "6.1 单帧 BEV 的物理局限：速度模糊与遮挡盲区", mode: ["full"] },
        { id: "ch06-sub02", title: "6.2 BEVDet4D：基于自车运动补偿的时序拼接", mode: ["full"] },
        { id: "ch06-sub03", title: "6.3 BEVStereo：利用前后帧双目几何构建代价体", mode: ["full"] }
      ]},
      { id: "ch07", number: "07", title: "BEVFormer：自顶向下注意力查询范式", subpages: [
        { id: "ch07-sub01", title: "7.1 从“自底向上抬升”到“自顶向下查询”的思想跃迁", mode: ["fast", "full"] },
        { id: "ch07-sub02", title: "7.2 BEV Query 网格与 3D 立柱参考点", mode: ["fast", "full"] },
        { id: "ch07-sub03", title: "7.3 空间交叉注意力（SCA）与多视角可形变采样", mode: ["fast", "full"] },
        { id: "ch07-sub04", title: "7.4 时序自注意力（TSA）与历史 BEV 循环流动", mode: ["full"] },
        { id: "ch07-sub05", title: "7.5 BEVFormer V1 ➔ V2：透视监督进化", mode: ["full"] }
      ]},
      { id: "ch08", number: "08", title: "Sparse4D：稠密网格向稀疏实例的演进", subpages: [
        { id: "ch08-sub01", title: "8.1 稠密 BEV 的算力浪费：90% 背景的立方级代价", mode: ["fast", "full"] },
        { id: "ch08-sub02", title: "8.2 900 稀疏 3D 锚框初始化机制", mode: ["fast", "full"] },
        { id: "ch08-sub03", title: "8.3 4D 关键点可形变聚合", mode: ["fast", "full"] },
        { id: "ch08-sub04", title: "8.4 Layer 0 ➔ Layer 6 逐层渐进式细化滑块演示", mode: ["fast", "full"] },
        { id: "ch08-sub05", title: "8.5 实例库（Instance Bank）的跨帧长期跟踪机制", mode: ["full"] }
      ]},
      { id: "ch09", number: "09", title: "Occupancy 基础：从 3D 框到通用体素空间", subpages: [
        { id: "ch09-sub01", title: "9.1 为什么 3D 边界框无法表征现实世界全部障碍物？", mode: ["fast", "full"] },
        { id: "ch09-sub02", title: "9.2 200x200x16 语义体素网格结构", mode: ["fast", "full"] },
        { id: "ch09-sub03", title: "9.3 Occ3D 真值格式与 17 种通用几何语义标签", mode: ["full"] },
        { id: "ch09-sub04", title: "9.4 评测指标革新：体素 mIoU 与视线射线 RayIoU", mode: ["full"] }
      ]},
      { id: "ch10", number: "10", title: "BEVDet-Occ：稠密 3D 占用网格基线", subpages: [
        { id: "ch10-sub01", title: "10.1 BEVDet-Occ 3D 视锥展开与反卷积网络架构", mode: ["full"] },
        { id: "ch10-sub02", title: "10.2 64 万体素的显存与算力瓶颈深入剖析", mode: ["full"] },
        { id: "ch10-sub03", title: "10.3 真实模型推断输出：地面 Z=0m 切片与全高度截面", mode: ["full"] }
      ]},
      { id: "ch11", number: "11", title: "SparseOcc：稀疏表面查询范式", subpages: [
        { id: "ch11-sub01", title: "11.1 稀疏表面的物理本质：空域 95% 自由空间的剔除", mode: ["full"] },
        { id: "ch11-sub02", title: "11.2 32,000 活跃表面体素粗到细掩码迭代机制", mode: ["full"] },
        { id: "ch11-sub03", title: "11.3 8 帧时序输入特征聚合与显存削减收益", mode: ["full"] },
        { id: "ch11-sub04", title: "11.4 真实模型推断输出与 Dense vs Sparse 对比", mode: ["full"] }
      ]},
      { id: "ch12", number: "12", title: "总结·工程对比·面试速查宝典", subpages: [
        { id: "ch12-sub01", title: "12.1 4 层严格审计实验性能总表", mode: ["interview", "fast", "full"] },
        { id: "ch12-sub02", title: "12.2 显存与延迟实测审计（显存占用 vs 权重文件大小辨析）", mode: ["interview", "fast", "full"] },
        { id: "ch12-sub03", title: "12.3 六大架构纵向演进全景路线图与工程权衡", mode: ["interview", "fast", "full"] },
        { id: "ch12-sub04", title: "12.4 自动驾驶算法岗 15 大高频深度面试题与标准答法", mode: ["interview", "fast", "full"] },
        { id: "ch12-sub05", title: "12.5 一句话精炼记忆卡 (Cheat Sheet)", mode: ["interview", "fast", "full"] },
        { id: "ch12-sub06", title: "12.6 全模型一键复现工程脚本指南", mode: ["interview", "full"] }
      ]}
    ]
  };
}

window.addEventListener('DOMContentLoaded', initApp);
window.toggleAccordion = toggleAccordion;
window.navigateTo = navigateTo;
