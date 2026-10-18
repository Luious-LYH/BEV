"""
Chapter 09: Occupancy 基础：从 3D 框到通用体素空间
Pedagogical and engineering deep-dive builder for subpages 9.1 to 9.4.
Compatible with Python 3.9+ (modular subpage functions, no nested f-strings).
"""

from scripts.builder.common import hero, card, how_to_read, interview_card, gotcha_box, tensor_flow_box, term_link, phenomenon_card

def build_ch09_sub01():
    h = hero("SCHEMATIC", "badge-schematic", "Closed-Set Bounding Boxes vs Open-Vocabulary 3D Voxels", 
             "9.1 为什么 3D 边界框无法表征现实世界全部障碍物？", 
             "直击自动驾驶最严峻的安全长尾痛点：长方体 3D 框为什么永远无法表达非刚性异形障碍物与非水平道路几何。")
    
    body = f"""
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            在自动驾驶领域，3D 目标检测（3D Bounding Box）主导了过去十年的感知研究。
            然而，当自动驾驶系统从封闭园区走向开放城市高阶辅助驾驶（NOA）时，3D 框的局限性暴露无遗：
        </p>

        <div class="how-to-read-box">
            <div class="how-to-read-title">🚨 传统 3D 边界框的三大致命软肋</div>
            <div class="guide-grid">
                <div class="guide-item" style="border-left-color: #EF4444;">
                    <div class="guide-item-title">1. 闭集分类的长尾盲区 (Closed-Set Curse)</div>
                    <div class="guide-item-text">
                        3D 检测依赖预先定义并标注好的有限类别（如 nuScenes 仅有 10 个类别）。
                        一旦道路上出现横躺在地的倒伏大树、高速公路遗撒的纸箱、侧翻的混凝土搅拌车或散落的轮胎皮，
                        传统 3D 检测网络会彻底“视而不见”，直接引发致命撞击事故！
                    </div>
                </div>
                <div class="guide-item" style="border-left-color: #F59E0B;">
                    <div class="guide-item-title">2. 刚性长方体的几何失真</div>
                    <div class="guide-item-text">
                        现实世界的障碍物绝非规则的砖块长方体。
                        带有机械臂的挖掘机、悬挂在空中的树枝、路旁弯曲的护栏和马路牙子，若用长方体去拟合，长方体内会包裹大量空地，导致下游规划控制误判发生过度保守刹车。
                    </div>
                </div>
                <div class="guide-item" style="border-left-color: #3B82F6;">
                    <div class="guide-item-title">3. 地表非平坦几何缺失</div>
                    <div class="guide-item-text">
                        3D Box 无法表达道路地表本身的起伏、坑洼（Pothole）和斜坡，而 {term_link('occupancy_concept', '3D Occupancy')} 提供了对连续三维环境的微元级完整刻画。
                    </div>
                </div>
            </div>
        </div>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>特性维度</th><th>传统 3D Bounding Box</th><th>3D Occupancy 体素占用网格</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>语义空间</b></td><td>预定义闭集（nuScenes 10 类，漏检未知类别）</td><td><b>开集通用几何（关注“物理空间是否存在实体”）</b></td></tr>
                    <tr><td><b>几何表现形式</b></td><td>7 自由度或 9 自由度刚性长方体 (x, y, z, w, l, h, roll, pitch, yaw)</td><td><b>三维欧几里得网格细分体素微元 (Voxel Grid)</b></td></tr>
                    <tr><td><b>长尾未知障碍物</b></td><td>完全失效（预测得分被 Softmax 归零）</td><td><b>完美防御（标记为 Occupied，输出物理防撞包络）</b></td></tr>
                    <tr><td><b>地表高程建模</b></td><td>无法建模（通常假设平面世界）</td><td><b>连续三维地形重建（支持坑洼、减速带、坡道）</b></td></tr>
                </tbody>
            </table>
        </div>
    """ + interview_card("面试官常问：既然特斯拉、华为等头部车企都发力 Occupancy，是不是意味着 3D Bounding Box 检测在未来会被彻底淘汰？", 
                         "【答题思路与融合架构辩证】：<br>"
                         "1. <b>明确结论：绝对不会淘汰，两者在未来将长期互补并存！</b><br>"
                         "2. <b>各司其职的系统分工</b>：<br>"
                         "   - <b>3D Bounding Box 的不可替代性</b>：对于标准交通参与者（汽车、行人），3D Box 提供了极其紧凑的 7 自由度参数和航向角速度矢量，便于下游行为预测大模型（Trajectory Prediction）理解物体的“行驶意图与交互博弈”；<br>"
                         "   - <b>3D Occupancy 的安全兜底价值</b>：Occupancy 不关心物体叫什么，只关心“那里有没有物理体积”。它为轨迹规划器提供最终的防碰撞物理凸包包络线，是防撞安全冗余的最后生命线。")

    c = card("从闭集 3D 检测到开集 3D 占用网格的必然演进", "SCHEMATIC", "badge-schematic", body)
    return f'<div id="subpage-ch09-sub01" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch09_sub02():
    h = hero("REAL DATA", "badge-real-data", "200x200x16 Semantic Voxel Grid Specification", 
             "9.2 200x200x16 语义体素网格结构", 
             "掌握业界主流 Occ3D 基准的网格空间划分与物理分辨率，体验基于真实 mini 数据集的交互式 3D 体素浏览器。")
    
    body = """
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>空间维度</th><th>物理覆盖范围 (米)</th><th>网格离散分辨率 (个体素)</th><th>单体素物理尺寸</th><th>物理几何含义</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>X 轴 (左右横向)</b></td><td>[-40.0m, +40.0m] (跨度 80m)</td><td>200 栅格</td><td><b>dx = 0.4 米</b></td><td>自车左右两侧各覆盖 40 米车道空间</td></tr>
                    <tr><td><b>Y 轴 (前后纵向)</b></td><td>[-40.0m, +40.0m] (跨度 80m)</td><td>200 栅格</td><td><b>dy = 0.4 米</b></td><td>自车前方 40 米与后方 40 米纵深</td></tr>
                    <tr><td><b>Z 轴 (垂直高度)</b></td><td>[-1.0m, +5.4m] (跨度 6.4m)</td><td>16 栅格</td><td><b>dz = 0.4 米</b></td><td>地下 1 米到车顶上方 5.4 米立体净空</td></tr>
                    <tr><td><b>全空间体素总量</b></td><td colspan="4"><b>200 × 200 × 16 = 640,000 个 3D 体素单元 (FP32 下每帧占 43.5MB 显存)</b></td></tr>
                </tbody>
            </table>
        </div>

        <h4 style="color: var(--accent-cyan); margin: 1.5rem 0 0.5rem;">🧱 真实 3D 语义占用体素交互式点云视检器 (Occ3D GT / BEVDet-Occ / SparseOcc)</h4>
        <div id="occ-metadata-banner" style="font-family: monospace; font-size: 0.8rem; color: var(--accent-green); margin-bottom: 0.5rem;"></div>

        <!-- 数据源与模型三模式动态切换 -->
        <div style="background: rgba(0, 229, 255, 0.05); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(0, 229, 255, 0.2); margin-bottom: 0.6rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
            <span style="font-size: 0.82rem; font-weight: 700; color: var(--accent-cyan);">📊 数据源切换:</span>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.loadRealData('gt'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #00E676;">🎯 Occ3D 真实真值 (GT 4,000 点)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.loadRealData('bevdet_occ'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}">🧱 BEVDet-Occ 稠密预测 (5,185 点)</button>
            <button class="btn-secondary active" onclick="if(occViewerInstance){occViewerInstance.loadRealData('sparseocc'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #00E5FF;">✨ SparseOcc 稀疏表面 (6,806 点)</button>
        </div>

        <!-- 语义颜色图例与说明 (解决红色/橙色等体素类别认知) -->
        <div style="background: rgba(10, 16, 28, 0.95); border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-cyan);">🎨 Occ3D 语义体素色彩图例速查 (点击颜色可快速单选过滤)：</span>
                <span style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">总计 17 种通用几何语义标签</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(0, 229, 255, 0.1); border: 1px solid #00E5FF; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #00E5FF; border-radius: 2px;"></span>
                    <b>青色：小型轿车 (Car)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(41, 121, 255, 0.1); border: 1px solid #2979FF; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #2979FF; border-radius: 2px;"></span>
                    <b>深蓝：卡车/挂车 (Truck/Trailer)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(255, 23, 68, 0.1); border: 1px solid #FF1744; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #FF1744; border-radius: 2px;"></span>
                    <b>🔴 红色：行人 / 交通锥 (Pedestrian / Cone)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(255, 171, 0, 0.1); border: 1px solid #FFAB00; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #FFAB00; border-radius: 2px;"></span>
                    <b>🟠 橙色/琥珀：工程车 / 水马护栏 (Construction / Barrier)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(168, 85, 247, 0.1); border: 1px solid #A855F7; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #A855F7; border-radius: 2px;"></span>
                    <b>🟣 紫色：沿街建筑物 / 人工结构 (Manmade / Building)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(0, 230, 118, 0.1); border: 1px solid #00E676; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #00E676; border-radius: 2px;"></span>
                    <b>🟩 绿色：行道树 / 灌木植被 (Vegetation / Terrain)</b>
                </span>
                <span class="legend-badge" style="display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(100, 116, 139, 0.15); border: 1px solid #64748B; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.78rem; color: #fff;">
                    <span style="width: 10px; height: 10px; background: #64748B; border-radius: 2px;"></span>
                    <b>⬛ 灰色：可通行沥青路面 / 人行道 (Drivable / Sidewalk)</b>
                </span>
            </div>
        </div>

        <!-- 视角预设与细粒度类别过滤 -->
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap; align-items: center;">
            <span style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted);">🎥 视角:</span>
            <button class="btn-secondary" onclick="if(occViewerInstance)occViewerInstance.resetView('3d')">🔄 3D 透视</button>
            <button class="btn-secondary" onclick="if(occViewerInstance)occViewerInstance.resetView('top')">⬇️ 俯视 (Top)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance)occViewerInstance.resetView('front')">➡️ 正视 (Front)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance)occViewerInstance.resetView('side')">⬆️ 侧视 (Side)</button>

            <span style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted); margin-left: 0.5rem;">🏷️ 交互分类过滤:</span>
            <button class="btn-secondary active" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('all'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}">🌐 全部</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('vehicles'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #00E5FF;">🚗 仅车辆 (Cyan/Blue)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('pedestrians_cones'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #FF1744;">🔴 行人/锥桶 (Red)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('barriers_construction'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #FFAB00;">🟠 护栏/障碍 (Orange)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('buildings'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #A855F7;">🟣 建筑物 (Purple)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('vegetation'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #10B981;">🌲 植被 (Green)</button>
            <button class="btn-secondary" onclick="if(occViewerInstance){occViewerInstance.setCategoryFilter('ground'); this.parentNode.querySelectorAll('button').forEach(b=>b.classList.remove('active')); this.classList.add('active');}" style="color: #94A3B8;">🛣️ 路面 (Gray)</button>
        </div>

        <div class="viewer-3d-box" style="height: 500px;">
            <canvas id="occ-canvas-3d" style="width: 100%; height: 100%;"></canvas>
        </div>
        <div id="occ-stats" style="font-family: monospace; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;"></div>
    """ + phenomenon_card(
        look_where="地面高度 Z=0 处的深色道路表面，以及自车周围由彩色体素构成的车辆与静态建筑物物理轮廓。",
        normal_pattern="真实 Occ3D GT 具有严密完整的物理轮廓，但仅覆盖相机视锥可见区；BEVDet-Occ 预测在物体边缘存在 1~2 格的卷积平滑膨胀；SparseOcc 呈现极度锐利的单体素薄壳结构，表面边界极为利落。",
        warning_signals="地表出现悬空体素（Z > 2m 处漂浮孤立点），或车辆内部出现大量实心死体素（光线无法穿透的物理矛盾区域）。",
        engineering_insight="纯视觉 Occupancy 通过多相机视线交汇求解几何逆问题，必须配合 RayIoU 评测以避免惩罚网络在完全不可见的视线盲区。",
        common_pitfall="误认为‘体素分辨率越高越好’——将 0.4m 细化到 0.2m 会使体素数量暴涨 8 倍（达到 512 万个体素），推理显存暴涨至 3GB+，导致车端 Orin-X 内存带宽瞬间打满窒息。",
        interview_30s="‘3D Occupancy 将感知从闭集 3D 边界框拓展到全场景微元级体素网格。Occ3D 采用 200x200x16 规格（0.4m 分辨率，64 万体素）。GT 仅在相机视线可见光线覆盖区有效；BEVDet-Occ 利用 3D 反卷积恢复稠密网格，而 SparseOcc 仅维护约 3.2 万活跃表面体素，显存直降近 70% 并天然支持长时序输入。’"
    ) + gotcha_box("体素分辨率 dx 与显存消耗的三次方权衡", 
                     "如果将体素分辨率从 0.4m 缩细到 0.2m，体素总数会暴增至 $400 \\times 400 \\times 32 = 5,120,000$（暴涨 8 倍）！反向传播时的显存开销将从 400MB 飙升至 3.2GB 以上。工程实践中必须在局部视野或采用多尺度金字塔以兼顾计算可行性。")

    c = card("Occ3D 空间网格物理参数规格表与 3D 交互视检器", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch09-sub02" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch09_sub03():
    h = hero("REAL DATA", "badge-real-data", "Occ3D 17 Geometric Semantic Categories & Voxel Labeling", 
             "9.3 Occ3D 真值格式与 17 种通用几何语义标签", 
             "深入 Occ3D 官方标注协议：类别 0 自由空间 (Free Space) 与 16 种真实障碍物物理定义。")
    
    body = """
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>标签 ID</th><th>语义类别名称 (Class Name)</th><th>物理占用属性</th><th>在真实场景中的体素占比</th><th>评测关注度</th></tr>
                </thead>
                <tbody>
                    <tr><td>0</td><td><b>free (自由空旷空间 / 空气)</b></td><td>空闲 (Free)</td><td><b>95.2% ~ 97.0% (极度占优)</b></td><td>计算 mIoU 时官方强制忽略该类</td></tr>
                    <tr><td>1</td><td>others (其他未归类杂物)</td><td>障碍物 (Occupied)</td><td>0.12%</td><td>关键长尾障碍物</td></tr>
                    <tr><td>2</td><td>barrier (施工挡板 / 护栏)</td><td>障碍物 (Occupied)</td><td>0.45%</td><td>防碰撞刚性实体</td></tr>
                    <tr><td>3</td><td>bicycle (自行车)</td><td>动态目标 (Occupied)</td><td>0.08%</td><td>弱势弱反小目标</td></tr>
                    <tr><td>4</td><td>bus (公共汽车 / 大巴)</td><td>动态目标 (Occupied)</td><td>0.35%</td><td>大体积高位遮挡源</td></tr>
                    <tr><td>5</td><td>car (普通小轿车)</td><td>动态目标 (Occupied)</td><td>0.85%</td><td>核心高频交通参与者</td></tr>
                    <tr><td>6</td><td>construction_vehicle (工程作业车)</td><td>动态目标 (Occupied)</td><td>0.15%</td><td>异形机械臂结构</td></tr>
                    <tr><td>7</td><td>motorcycle (摩托车)</td><td>动态目标 (Occupied)</td><td>0.09%</td><td>高机动快速目标</td></tr>
                    <tr><td>8</td><td>pedestrian (行人)</td><td>动态目标 (Occupied)</td><td>0.05%</td><td>最高安全优先级保护对象</td></tr>
                    <tr><td>9</td><td>traffic_cone (交通路锥)</td><td>小障碍物 (Occupied)</td><td>0.03%</td><td>贴地小体积目标</td></tr>
                    <tr><td>10</td><td>trailer (挂车)</td><td>动态目标 (Occupied)</td><td>0.22%</td><td>超长多铰接车身</td></tr>
                    <tr><td>11</td><td>truck (卡车)</td><td>动态目标 (Occupied)</td><td>0.41%</td><td>高位盲区车辆</td></tr>
                    <tr><td>12</td><td>driveable_surface (可行驶路面)</td><td>地表 (Occupied)</td><td>2.10%</td><td>规划通行走廊基准</td></tr>
                    <tr><td>13</td><td>other_flat (其他平坦地表)</td><td>地表 (Occupied)</td><td>0.31%</td><td>路边人行横道/绿化带边缘</td></tr>
                    <tr><td>14</td><td>sidewalk (人行道)</td><td>路缘 (Occupied)</td><td>0.65%</td><td>路缘石高度跃迁</td></tr>
                    <tr><td>15</td><td>terrain (起伏地形/草地)</td><td>地形 (Occupied)</td><td>0.48%</td><td>非结构化不可通行地表</td></tr>
                    <tr><td>16</td><td>manmade (建筑物/立柱/围墙)</td><td>静态障碍 (Occupied)</td><td>0.72%</td><td>全景背景刚性结构</td></tr>
                </tbody>
            </table>
        </div>

        <div class="action-btn-row" style="margin-top: 1rem;">
            <button class="btn-secondary" onclick="openCodeModal('occupancy_metrics')">💻 查看 Occ3D 官方 mIoU 评测核心计算实现</button>
        </div>
    """ + interview_card("面试官常问：Occ3D 中 95% 以上都是 Free 空间，如果直接用普通的交叉熵 Cross-Entropy 训练会发生什么？如何解决严重的正负样本失衡？", 
                         "【损失函数设计与类别平衡】：<br>"
                         "1. <b>直接 CE 训练的崩盘结局</b>：网络会瞬间过拟合‘将所有体素都预测为 Free’的偷懒捷径，准确率高达 96%，但有价值的物体检出率彻底降为 0！<br>"
                         "2. <b>工业界三大解决方案</b>：<br>"
                         "   - <b>加权 Focal Loss</b>：对 Free 类别的权重压低为 0.05，大幅增强稀缺障碍物体素的梯度贡献；<br>"
                         "   - <b>Lovász-Softmax Loss</b>：直接优化连续近似的 IoU 指标，对样本不平衡具备天然的鲁棒性；<br>"
                         "   - <b>可见光线深度一致性重加权 (Ray-Loss)</b>：强化光线穿过的前沿表面体素，弱化背景空气的监督。")

    c = card("Occ3D 17 类别语义标签定义表", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch09-sub03" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch09_sub04():
    h = hero("SCHEMATIC", "badge-schematic", "Voxel mIoU vs Occlusion-Aware RayIoU Metrics", 
             "9.4 评测指标革新：体素 mIoU 与视线射线 RayIoU", 
             "剖析新一代 {term_link('ray_iou_concept', 'RayIoU')} 评测指标：为什么传统 Voxel mIoU 逼迫模型在完全不可见的视线盲区去“盲猜真值”，以及如何公正评价纯视觉占用网络。")
    
    body = f"""
        <!-- 官方核心评测图：RayIoU 视线光线遮挡物理建模图 -->
        <div class="media-container" style="margin-bottom: 1.5rem;">
            <img src="learning_report/images/paper_architectures/sparseocc_rayiou.jpg" alt="RayIoU Evaluation Principle Diagram" onclick="openLightbox(this.src, 'RayIoU 官方论文评测机理图：视线追踪光线穿透与首次击中物理模型', 'CORE EVALUATION', 'badge-real-data')">
            <div class="media-caption">
                📐 <b>RayIoU 视线射线遮挡感知评测机理 (CVPR 2024)</b>：模拟光子自相机光心向三维空间沿射线追踪（Ray-Casting）。在射线击中第一个障碍物表面之前累加 Free 空间一致性；在表面位置检验障碍物分类；障碍物背后的不可见盲区直接终止射线，不产生误罚。
            </div>
        </div>

        <div style="background: rgba(10, 14, 23, 0.9); padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 1.5rem;">
            <h4 style="color: var(--accent-cyan); margin-bottom: 0.5rem;">1. 传统全局体素交并比 (Voxel mIoU)</h4>
            <div style="background: #020408; padding: 0.85rem 1.1rem; border-radius: 6px; border: 1px solid rgba(0,229,255,0.2); margin-bottom: 1rem; overflow-x: auto;">
                $$\\text{{IoU}}_c = \\frac{{\\text{{TP}}_c}}{{\\text{{TP}}_c + \\text{{FP}}_c + \\text{{FN}}_c}}, \\quad \\text{{mIoU}} = \\frac{{1}}{{16}} \\sum_{{c=1}}^{{16}} \\text{{IoU}}_c$$
            </div>

            <h4 style="color: var(--accent-purple); margin: 1.25rem 0 0.5rem;">2. 视线射线遮挡感知交并比 (RayIoU, CVPR 2024)</h4>
            <div style="background: #020408; padding: 0.85rem 1.1rem; border-radius: 6px; border: 1px solid rgba(139,92,246,0.2); margin-bottom: 1rem; overflow-x: auto;">
                $$\\text{{RayIoU}} = \\frac{{1}}{{|\\mathcal{{R}}|}} \\sum_{{r \\in \\mathcal{{R}}}} \\frac{{\\sum_d \\text{{TP}}(r, d)}}{{\\sum_d \\left[ \\text{{TP}}(r, d) + \\text{{FP}}(r, d) + \\text{{FN}}(r, d) \\right]}}$$
                $$d^*(r) = \\min \\{{ d \\mid \\mathbf{{V}}(r(d)) \\neq \\text{{Free}} \\}} \\quad (\\text{{射线打在物体表面后立即终止，不评估盲区}})$$
            </div>
        </div>

        <div class="how-to-read-box">
            <div class="how-to-read-title">📊 评测维度的物理与工程认知</div>
            <div class="guide-grid">
                <div class="guide-item" style="border-left-color: #EF4444;">
                    <div class="guide-item-title">Voxel mIoU 的“不可见惩罚”</div>
                    <div class="guide-item-text">
                        在激光雷达或密集点云累积真值中，建筑物后方的几何可能被标注出来了，但相机视角在物理上被阻挡。Voxel mIoU 强行惩罚相机未预测出遮挡后的空间，导致指标无法客观反映感知模型的真实视觉能力。
                    </div>
                </div>
                <div class="guide-item" style="border-left-color: #10B981;">
                    <div class="guide-item-title">RayIoU 的“前视一致性”</div>
                    <div class="guide-item-text">
                        RayIoU 模拟激光雷达从自车中心发射射线，沿着视线方向考察最近障碍物表面的几何命中情况。如果模型正确预测了物体正面轮廓，即使背面不可见也不会遭受错误惩罚。
                    </div>
                </div>
            </div>
        </div>
    """ + interview_card("面试官常问：RayIoU 相比传统 Voxel mIoU，核心解决了什么评估痛点？在工业实车评测中为什么越来越推崇 RayIoU？", 
                         "【评测公正性与物理本质】：<br>"
                         "1. <b>剔除看不见的盲区不公正惩罚</b>：在真实道路上，大货车身后的体素完全被遮挡，相机在物理上接收不到任何光子。传统 Voxel mIoU 却把盲区全部计入 FN 惩罚，迫使模型去训练集上过拟合盲区的‘幻觉先验’。<br>"
                         "2. <b>精准聚焦车辆安全视距</b>：RayIoU 沿着相机中心向外发射数千条光线，仅考察【光线在遇到第一个障碍物表面之前】的物理可见段。这与自动驾驶车辆控制系统的避障需求 100% 吻合：控制算法只需要知道‘沿着当前行车光线，最近的障碍物表面在哪里’，而不关心障碍物背后隐藏的不可见空间。")

    c = card("Voxel mIoU 与 RayIoU 数学定义深度对比", "SCHEMATIC", "badge-schematic", body)
    return f'<div id="subpage-ch09-sub04" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch09():
    return "\n".join([
        build_ch09_sub01(),
        build_ch09_sub02(),
        build_ch09_sub03(),
        build_ch09_sub04(),
    ])
