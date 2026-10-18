"""
Chapter 02: 坐标与几何：从 3D 物理世界到 2D 像素
Pedagogical and engineering deep-dive builder for subpages 2.1 to 2.4.
Compatible with Python 3.9+ (modular subpage functions, no nested f-strings).
"""

from scripts.builder.common import hero, card, how_to_read, interview_card, gotcha_box, tensor_flow_box, term_link

def build_ch02_sub01():
    h = hero("REAL DATA", "badge-real-data", "Sensor ➔ Ego ➔ Global ➔ Camera Coordinates Topology", 
             "2.1 自动驾驶四大坐标系深度剖析", 
             "掌握自动驾驶感知最底层的空间几何四级基准：激光雷达系、自车后轴系、高精世界地图系与相机透视系的原点、手性规则与拓扑闭环。")
    
    body1 = """
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            在自动驾驶多传感器融合系统中，任何一个没有指明<b>参考坐标系（Reference Frame）</b>的 3D 坐标都是毫无物理意义的危险数字。
            不同传感器由于工业标准（如 ISO 8855 车辆动力学标准 vs OpenCV 计算机视觉标准）不同，其坐标系原点与轴向定义存在本质差异：
        </p>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>坐标系名称</th><th>原点位置 (Origin)</th><th>X 轴朝向</th><th>Y 轴朝向</th><th>Z 轴朝向</th><th>手性规则</th><th>核心承载感知任务</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>激光雷达系 (LiDAR)</b></td><td>车顶雷达光学发射中心</td><td>车头正右方</td><td>车头正前方</td><td>天顶正上方</td><td>右手系</td><td>点云原始存储、反射率测量</td></tr>
                    <tr><td><b>自车系 (Ego Frame)</b></td><td>后轴中心在地面垂直投影</td><td>车辆正右方</td><td>车辆正前方</td><td>天顶正上方</td><td>右手系</td><td><b>BEV 鸟瞰特征网格、3D 检测真值</b></td></tr>
                    <tr><td><b>世界系 (Global Frame)</b></td><td>高精地图固定绝对锚点</td><td>地理正东 (East)</td><td>地理正北 (North)</td><td>地心法线向上</td><td>右手系</td><td>跨时序位姿对齐、SLAM 全局建图</td></tr>
                    <tr><td><b>相机系 (Camera Frame)</b></td><td>镜头感光芯片光学中心</td><td>图像正右方</td><td>图像正下方</td><td><b>视轴前方 (光轴深度)</b></td><td>右手系 (OpenCV)</td><td>透视投影、单目深度估计、Backbone 2D 特征抽取</td></tr>
                </tbody>
            </table>
        </div>
    """
    c1 = card("四大坐标系几何定义与标准规范对比", "REAL DATA", "badge-real-data", body1)

    svg_chain = """
        <div style="background: #060913; border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 8px; padding: 1.5rem 1rem; margin-bottom: 1.5rem; overflow-x: auto;">
            <svg viewBox="0 0 920 380" style="width: 100%; min-width: 800px; height: auto;" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <marker id="arrowCyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 1 L 10 5 L 0 9 z" fill="#00E5FF"/>
                    </marker>
                    <marker id="arrowGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 1 L 10 5 L 0 9 z" fill="#00E676"/>
                    </marker>
                    <marker id="arrowPurple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 1 L 10 5 L 0 9 z" fill="#A855F7"/>
                    </marker>
                    <marker id="arrowOrange" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 1 L 10 5 L 0 9 z" fill="#FF9100"/>
                    </marker>
                </defs>

                <g transform="translate(140, 80)">
                    <rect x="-85" y="-35" width="170" height="70" rx="8" fill="#0A181C" stroke="#00E676" stroke-width="2"/>
                    <text x="0" y="-8" fill="#00E676" font-size="13" font-weight="bold" text-anchor="middle">LiDAR 坐标系</text>
                    <text x="0" y="14" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">P_lidar = (X, Y, Z)</text>
                </g>

                <g transform="translate(460, 80)">
                    <rect x="-100" y="-40" width="200" height="80" rx="8" fill="#051B2C" stroke="#00E5FF" stroke-width="2.5"/>
                    <text x="0" y="-12" fill="#00E5FF" font-size="14" font-weight="bold" text-anchor="middle">自车系 (Ego Frame)</text>
                    <text x="0" y="8" fill="#E2E8F0" font-size="11" font-weight="bold" text-anchor="middle">★ BEV 网格统一原点</text>
                    <text x="0" y="26" fill="#94A3B8" font-size="9" font-family="monospace" text-anchor="middle">P_ego = (X_ego, Y_ego, Z_ego)</text>
                </g>

                <line x1="225" y1="80" x2="350" y2="80" stroke="#00E676" stroke-width="2.5" marker-end="url(#arrowGreen)"/>
                <text x="288" y="70" fill="#00E676" font-size="10" font-family="monospace" text-anchor="middle">T_l2e: R_l2e, t_l2e</text>

                <g transform="translate(780, 80)">
                    <rect x="-85" y="-35" width="170" height="70" rx="8" fill="#1A0D2E" stroke="#A855F7" stroke-width="2"/>
                    <text x="0" y="-8" fill="#A855F7" font-size="13" font-weight="bold" text-anchor="middle">世界系 (Global)</text>
                    <text x="0" y="14" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">高精地图高程锚点</text>
                </g>

                <path d="M 560 65 L 685 65" stroke="#A855F7" stroke-width="2" marker-end="url(#arrowPurple)"/>
                <text x="622" y="55" fill="#A855F7" font-size="10" font-family="monospace" text-anchor="middle">T_e2g (当前位姿)</text>

                <path d="M 685 95 L 560 95" stroke="#A855F7" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrowPurple)"/>
                <text x="622" y="112" fill="#A855F7" font-size="9" font-family="monospace" text-anchor="middle">T_g2e (逆向变换)</text>

                <g transform="translate(460, 240)">
                    <rect x="-100" y="-35" width="200" height="70" rx="8" fill="#241407" stroke="#FF9100" stroke-width="2"/>
                    <text x="0" y="-8" fill="#FF9100" font-size="13" font-weight="bold" text-anchor="middle">相机光学系 (Camera)</text>
                    <text x="0" y="12" fill="#FED7AA" font-size="10" font-family="monospace" text-anchor="middle">Z_cam: 前向深度 (Depth)</text>
                </g>

                <line x1="460" y1="120" x2="460" y2="195" stroke="#FF9100" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
                <text x="475" y="160" fill="#FF9100" font-size="10" font-family="monospace" text-anchor="start">T_e2c = (T_c2e)^(-1)</text>

                <g transform="translate(780, 240)">
                    <rect x="-85" y="-35" width="170" height="70" rx="8" fill="#260C1A" stroke="#F43F5E" stroke-width="2"/>
                    <text x="0" y="-8" fill="#F43F5E" font-size="13" font-weight="bold" text-anchor="middle">像素坐标系 (Pixel 2D)</text>
                    <text x="0" y="14" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">(u, v) ∈ [0,1600]x[0,900]</text>
                </g>

                <line x1="560" y1="240" x2="685" y2="240" stroke="#F43F5E" stroke-width="2.5" stroke-dasharray="5,4" marker-end="url(#arrowCyan)"/>
                <text x="622" y="230" fill="#F43F5E" font-size="10" font-family="monospace" text-anchor="middle">透视投影 K [I|0]</text>
                <text x="622" y="260" fill="#64748B" font-size="9" text-anchor="middle">(不可逆透视除法)</text>

                <text x="140" y="340" fill="#64748B" font-size="11">● 实线: 3D 刚体外参变换 SE(3) (长度角度严格保真)</text>
                <text x="560" y="340" fill="#64748B" font-size="11">● 虚线: 针孔透视投影 (三维压二维，丢失绝对深度)</text>
            </svg>
        </div>
    """

    body2 = """
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            下方为<b>原生暗色高对比度 SVG 坐标拓扑网络</b>：
            清晰展示从原始激光点到自车系、世界系，再逆向变换至相机系并最终投射到 2D 像素平面的严密数学闭环链条。
        </p>
    """ + svg_chain + """
        <div style="background: rgba(0,0,0,0.4); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 1rem 1.25rem;">
            <div style="font-size: 0.88rem; font-weight: 600; color: var(--accent-cyan); margin-bottom: 0.5rem;">
                📋 真实外参矩阵数值解剖 (CAM_FRONT ➔ Ego Frame，来自 scene-0061 真实校准文件)
            </div>
            <div style="font-family: monospace; font-size: 0.8rem; color: #E2E8F0; line-height: 1.6;">
                位移向量 t_c2e (米): <code>[1.7008, 0.0159, 1.5109]</code> (前轴上方偏右 1.59cm，离地 1.51m)<br>
                旋转四元数 q_c2e: <code>[0.4998, -0.5030, 0.4998, -0.4974]</code> (转换为旋转矩阵 SO(3) 保持 det=1.0)
            </div>
        </div>
    """ + interview_card("面试官常问：为什么激光雷达点投影到相机图像上时，必须先转到自车系 (Ego)，而不是直接求一个从 LiDAR 到 Camera 的外参矩阵 T_l2c？", 
                         "【答题思路与工程架构灵活性】：<br>"
                         "1. <b>模块解耦与 $O(N)$ 复杂度优势</b>：若每两个传感器之间都直接标定一个相对外参，对于 1 激光 + 6 相机 + 5 毫米波共 12 个传感器，需要维护标定矩阵数量高达 $C_{12}^2 = 66$ 个！一旦车顶激光雷达发生更换，需要重新标定与其他 11 个传感器的所有矩阵。若以统一自车系为星型拓扑中枢，只需维护 12 个 $T_{\\text{sensor}\\to\\text{ego}}$ 矩阵，系统复杂度从 $O(N^2)$ 骤降至 $O(N)$。<br>"
                         "2. <b>时序位姿与多帧对齐统一基准</b>：自车运动补偿必须在自车系进行。将点云转到自车系后，可直接结合自车里程计位姿 $T_{\\text{ego}(t) \\to \\text{global}}$ 参与多帧累积，保持底层几何管线的绝对正交与规整。")

    c2 = card("四大坐标系刚体外参变换闭环拓扑网络", "NATIVE DARK SVG", "badge-interactive", body2)
    return f'<div id="subpage-ch02-sub01" class="subpage-content-section" style="display:none;">{h}\n{c1}\n{c2}</div>'

def build_ch02_sub02():
    h = hero("REAL DATA", "badge-real-data", "Rigid Body SE(3) Transform & Pin-Hole KaTeX Formulations", 
             "2.2 外参刚体变换与内参透视投影公式推导", 
             "推导自动驾驶感知几何的严密数学公式：从旋转四元数到 SO(3) 正交矩阵、SE(3) 刚体运动，再到针孔相机透视投影与深度多义性。")
    
    body = """
        <div style="background: rgba(10, 14, 23, 0.9); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-subtle); margin-bottom: 1.5rem;">
            <h4 style="color: var(--accent-cyan); margin-bottom: 0.6rem; font-size: 1.05rem;">
                1. 刚体欧式外参变换 (Rigid Body Transformation in SE(3))
            </h4>
            <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.75rem;">
                刚体变换不发生任何形变、拉伸或缩放，由旋转矩阵 $R \\in SO(3)$（度量 Roll, Pitch, Yaw）与平移向量 $t \\in \\mathbb{R}^3$ 构成：
            </p>
            <div style="background: #020408; padding: 1rem 1.25rem; border-radius: 6px; border: 1px solid rgba(0,229,255,0.2); margin-bottom: 1rem; overflow-x: auto;">
                $$P_{\\text{dst}} = R \\cdot P_{\\text{src}} + t = \\begin{bmatrix} R & t \\\\ 0_{1\\times 3} & 1 \\end{bmatrix} \\begin{bmatrix} P_{\\text{src}} \\\\ 1 \\end{bmatrix} \\in SE(3)$$
                $$R^T R = I, \\quad \\det(R) = +1$$
            </div>

            <h4 style="color: var(--accent-purple); margin: 1.25rem 0 0.6rem; font-size: 1.05rem;">
                2. 四元数到旋转矩阵转化 (Quaternion to SO(3))
            </h4>
            <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.75rem;">
                标定参数中给出的单位四元数 $q = [w, x, y, z]^T$（满足 $w^2 + x^2 + y^2 + z^2 = 1$），转换为 $3\\times 3$ 正交旋转矩阵：
            </p>
            <div style="background: #020408; padding: 1rem 1.25rem; border-radius: 6px; border: 1px solid rgba(139,92,246,0.2); margin-bottom: 1rem; overflow-x: auto;">
                $$R(q) = \\begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\\\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\\\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \\end{bmatrix}$$
            </div>

            <h4 style="color: var(--accent-orange); margin: 1.25rem 0 0.6rem; font-size: 1.05rem;">
                3. 针孔相机透视投影方程 (Pinhole Perspective Projection)
            </h4>
            <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.75rem;">
                相机坐标系中的 3D 点 $P_{\\text{cam}} = [X_c, Y_c, Z_c]^T$ 经过内参矩阵 $K$ 透视投影到像素平面 $(u, v)$：
            </p>
            <div style="background: #020408; padding: 1rem 1.25rem; border-radius: 6px; border: 1px solid rgba(255,145,0,0.2); margin-bottom: 1rem; overflow-x: auto;">
                $$s \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} = K \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\end{bmatrix} = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} X_c \\\\ Y_c \\\\ Z_c \\end{bmatrix}$$
                $$u = f_x \\frac{X_c}{Z_c} + c_x, \\quad v = f_y \\frac{Y_c}{Z_c} + c_y, \\quad s = Z_c \\quad (\\text{物理光轴深度 Depth})$$
            </div>
        </div>
    """ + gotcha_box("透视投影除以深度 Zc 的‘除零与负数’数学陷阱", 
                    "当 3D 点落在相机镜头后方时，$Z_c \\le 0$。若直接应用透视公式，不仅会导致除以 0 产生 NaN，还会导致位于后方的点在除以负数后‘镜像翻转’投影到前方图像画面中！在做点云或 3D Bounding Box 投影时，必须设置严格的近平面几何截断：<code>if pt_cam[2] <= 0.1: continue</code>。"
    ) + interview_card("面试官常问：为什么说 2D 像素反投影回 3D 空间是病态的 (Ill-Posed)？LSS 与单目 3D 检测分别是如何解决这一深度多义性的？", 
                       "【答题思路与本质剖析】：<br>"
                       "1. <b>深度的不可逆丢失</b>：从针孔模型可知，像素 $(u, v)$ 仅仅锁定了空间射线上的单位方向向量 $[(u-c_x)/f_x, (v-c_y)/f_y, 1]^T$。同一个像素可能对应 1 米外的乒乓球，也可能对应 100 米外的一轮满月（尺度与深度不可分辨性）。<br>"
                       "2. <b>LSS 解法（离散深度概率分布）</b>：不强求回归唯一确定深度，而是预测沿视线 118 个深度区间的概率分布 $P(d|u,v)$，将特征沿视锥广播外积展开。<br>"
                       "3. <b>单目 3D 检测解法（几何语义先验）</b>：依靠物体类别先验（如轿车车身通常高 1.5m），利用 $h_{2D} \\approx f_y \\cdot \\frac{H_{3D}}{Z}$ 逆向推算中心深度，但极易受到地表起伏与视距遮挡的干扰。")

    c = card("核心几何数学推导 (KaTeX 完整渲染)", "MATHEMATICAL RIGOR", "badge-real-data", body)
    return f'<div id="subpage-ch02-sub02" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch02_sub03():
    h = hero("REAL DATA", "badge-real-data", "Step-by-Step Matrix Transform Animation Simulator", 
             "2.3 真实 LiDAR 点投影到相机：分步动画模拟器", 
             "跟随来自 nuScenes 真实标定文件的激光雷达点，交互式验证 4 步刚体外参变换与针孔相机投影，观察视锥内外有效性判定。")
    
    body = """
        <div style="background: rgba(10, 16, 28, 0.95); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: var(--accent-cyan); font-size: 0.95rem; margin-bottom: 0.75rem;">
                🎯 第一步：选择待投影的真实激光雷达点 (来自 nuScenes scene-0061 真实点云)
            </div>
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
                <button id="coord-point-btn-0" class="btn-secondary active" onclick="selectCoordPoint(0)">🎯 测试点 1: 视野正中 (前车车牌)</button>
                <button id="coord-point-btn-1" class="btn-secondary" onclick="selectCoordPoint(1)">⚠️ 测试点 2: 图像右侧近界 (路缘石)</button>
                <button id="coord-point-btn-2" class="btn-secondary" onclick="selectCoordPoint(2)">🚫 测试点 3: 车后盲区 (Z_cam &lt; 0)</button>
            </div>

            <div style="font-weight: 700; color: var(--accent-green); font-size: 0.95rem; margin-bottom: 0.75rem;">
                🔄 第二步：单步单矩阵推导演练 (逐步查看齐次矩阵运算)
            </div>
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
                <button id="coord-step-btn-1" class="btn-primary active" onclick="setCoordStep(1)">步骤 1: 雷达系 ➔ 自车系</button>
                <button id="coord-step-btn-2" class="btn-secondary" onclick="setCoordStep(2)">步骤 2: 自车系 ➔ 相机系</button>
                <button id="coord-step-btn-3" class="btn-secondary" onclick="setCoordStep(3)">步骤 3: 归一化平面</button>
                <button id="coord-step-btn-4" class="btn-secondary" onclick="setCoordStep(4)">步骤 4: 内参投影像素</button>
            </div>

            <div id="coord-warn-banner" style="display: none; background: rgba(244, 63, 94, 0.15); border: 1px solid #f43f5e; color: #fecaca; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-weight: 600;"></div>

            <!-- 当前步骤动态解说卡片 -->
            <div style="background: #020408; border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <h4 id="coord-sim-title" style="color: var(--accent-cyan); margin-bottom: 0.5rem; font-size: 1.05rem;">测试点 1：视野正中央前景目标 (前车保险杠) - 第一步：激光雷达原生点 P_lidar</h4>
                <p id="coord-sim-desc" style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.75rem;">在车顶激光雷达坐标系下读取激光脉冲回波点。</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 0.75rem 1rem; border-radius: 6px;">
                        <div style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">当前三维向量 / 像素坐标</div>
                        <div id="coord-sim-vector" style="font-family: monospace; font-size: 0.95rem; color: #00E676; font-weight: 700; margin-top: 0.25rem;">P_lidar = [-0.452, 25.379, -1.184] (m)</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 0.75rem 1rem; border-radius: 6px;">
                        <div style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">所用标定参数 / 转换矩阵</div>
                        <div id="coord-sim-matrix" style="font-family: monospace; font-size: 0.95rem; color: #38BDF8; font-weight: 700; margin-top: 0.25rem;">Frame: LIDAR_TOP (X-Right, Y-Forward, Z-Up)</div>
                    </div>
                </div>

                <div style="background: rgba(0, 229, 255, 0.05); border-left: 3px solid var(--accent-cyan); padding: 0.6rem 0.9rem; border-radius: 4px;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: var(--accent-cyan); margin-bottom: 0.2rem;">💡 工程判定与物理含义：</div>
                    <div id="coord-sim-status" style="font-size: 0.85rem; color: var(--text-primary); line-height: 1.5;">原生空间：自车中线左偏 0.45m，前方 25.38m，高度低于雷达 1.18m</div>
                </div>
            </div>

            <!-- 2D 模拟前视相机画面与准星指示 -->
            <div style="margin-bottom: 1.25rem;">
                <div style="font-weight: 700; color: var(--text-primary); font-size: 0.9rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <span>📷 前视相机 (CAM_FRONT 1600×900) 成像投影指示器</span>
                    <span style="font-size: 0.78rem; color: var(--text-muted); font-family: monospace;">[u: 0~1600px, v: 0~900px]</span>
                </div>
                <div id="coord-canvas-wrapper" style="position: relative; width: 100%; aspect-ratio: 16/9; max-height: 400px; background: #070d18; border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 8px; overflow: hidden; background-image: radial-gradient(rgba(0, 229, 255, 0.15) 1px, transparent 1px); background-size: 32px 32px;">
                    <!-- 视野网格十字中心参考线 -->
                    <div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(255, 255, 255, 0.1);"></div>
                    <div style="position: absolute; top: 50%; left: 0; right: 0; height: 1px; background: rgba(255, 255, 255, 0.1);"></div>
                    
                    <!-- 光学中心标定标记 (cx=816, cy=491) -->
                    <div style="position: absolute; left: 51%; top: 54.5%; font-size: 0.7rem; color: rgba(255,255,255,0.4); font-family: monospace;">
                        + 主点 Principal Point (cx, cy)
                    </div>

                    <!-- 动态目标点准星 -->
                    <div id="coord-crosshair" style="display: none; position: absolute; width: 28px; height: 28px; transform: translate(-50%, -50%); pointer-events: none; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);">
                        <div style="position: absolute; left: 0; top: 0; right: 0; bottom: 0; border: 2px solid #00E5FF; border-radius: 50%; box-shadow: 0 0 12px rgba(0,229,255,0.8); animation: pulse 1.5s infinite;"></div>
                        <div style="position: absolute; left: 50%; top: 32px; transform: translateX(-50%); background: rgba(0,0,0,0.85); color: #00E5FF; font-family: monospace; font-size: 0.72rem; padding: 2px 6px; border-radius: 3px; white-space: nowrap; border: 1px solid rgba(0,229,255,0.4);">
                            Target Pixel (u, v)
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="action-btn-row" style="margin-top: 1rem;">
            <button class="btn-secondary" onclick="openCodeModal('coordinate_projection')">💻 查看官方齐次矩阵投影代码 (Python/PyTorch)</button>
        </div>
    """
    c = card("真实激光点投影全流程分步模拟器", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch02-sub03" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch02_sub04():
    h = hero("REAL DATA", "badge-real-data", "3D Bounding Box 8-Corner Camera Reprojection", 
             "2.4 3D 边界框（Bounding Box）相机视角重投影", 
             "掌握 3D 边界框 7 自由度参数定义、8 顶点空间几何解算以及在 6 路环视透视图像上的线框投影渲染。")
    
    body = """
        <div class="media-container">
            <img src="learning_report/images/real/nuscenes/04_gt_3d_boxes_on_cameras.png" alt="3D Boxes on Cameras" onclick="openLightbox(this.src, '3D 边界框投射到 6 路环视相机', 'REAL DATA', 'badge-real-data')">
            <div class="media-caption">
                📦 真实 3D 边界框投射在 6 路环视图像：青色线框代表车辆，黄色为行人，红色箭头标注车辆航向角 (Yaw) 朝向。
            </div>
        </div>

        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            自动驾驶中，标注真值是一个包含 7 个物理自由度的 3D 边界框：
            中心坐标 $(x, y, z)$、几何尺寸 $(w, l, h)$ 以及偏航航向角 $\\theta$ (Yaw)。
            要将一个 3D 框在相机图像上画出来，必须先在自车空间解算出长方体的 8 个物理顶点坐标，再将每个顶点通过外参内参透视投影到 2D 像素平面，并按 12 条棱边连线。
        </p>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>顶点编号</th><th>局部几何坐标 (未旋转前)</th><th>物理几何方位</th></tr>
                </thead>
                <tbody>
                    <tr><td>Corner 0</td><td>(+l/2, +w/2, -h/2)</td><td>车头右下角</td></tr>
                    <tr><td>Corner 1</td><td>(+l/2, -w/2, -h/2)</td><td>车头左下角</td></tr>
                    <tr><td>Corner 2</td><td>(-l/2, -w/2, -h/2)</td><td>车尾左下角</td></tr>
                    <tr><td>Corner 3</td><td>(-l/2, +w/2, -h/2)</td><td>车尾右下角</td></tr>
                    <tr><td>Corner 4~7</td><td>(+/- l/2, +/- w/2, +h/2)</td><td>车顶对应的 4 个高位顶点</td></tr>
                </tbody>
            </table>
        </div>
    """ + interview_card("面试官常问：在相机边缘截断（Truncation）时，部分 3D 边界框的顶点跑到相机后方 (Z_cam <= 0) 或超出图像边界，算法在可视化与 2D-3D 关联时如何做视锥近裁切 (Near Clipping)？", 
                         "【答题思路与图形学算法】：<br>"
                         "1. <b>直接投影会导致画面崩坏</b>：若一个长 12 米的大卡车靠近自车，车头在相机前，车尾在相机后。直接投影车尾顶点会产生负无穷像素坐标，画出的线框会撕裂全图。<br>"
                         "2. <b>图形学 Sutherland-Hodgman 视锥线段裁剪</b>：在相机坐标系下，于近裁切平面（如 $Z = 0.2\\text{m}$）处建立截断平面。对于穿过近平面的 3D 棱边，计算棱边线段与 $Z=0.2$ 平面的几何交点，生成新的截断顶点，仅将位于截平面正前方的新线段投射到图像上。")

    c = card("3D 边界框投射到 6 路环视相机 (Ground Truth Overlays)", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch02-sub04" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch02():
    return "\n".join([
        build_ch02_sub01(),
        build_ch02_sub02(),
        build_ch02_sub03(),
        build_ch02_sub04()
    ])
