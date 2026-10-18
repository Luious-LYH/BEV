"""
Chapter 03: BEV 基础：为什么自动驾驶需要鸟瞰图？
Pedagogical and engineering deep-dive builder for subpages 3.1 to 3.3.
Compatible with Python 3.9+ (modular subpage functions, no nested f-strings).
"""

from scripts.builder.common import hero, card, how_to_read, interview_card, gotcha_box, tensor_flow_box, term_link, phenomenon_card

def build_ch03_sub01():
    h = hero("SCHEMATIC", "badge-schematic", "Perspective View vs Bird's-Eye View Fundamental Limits", 
             "3.1 透视视角的深度缺失与尺度畸变", 
             "深刻剖析传统 2D 透视图像在空间感知中的物理硬伤：尺度不一致、严重前后遮挡、无法直接服务规控。")
    
    body = """
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            在自动驾驶感知发展早期，业界普遍采用“单目/多目 2D 目标检测 + 启发式几何回归深度”的二阶段方案。
            但随着车辆向高速领航（NOA）和复杂城市场景演进，2D 透视视角的物理缺陷成为制约算法安全性的首要瓶颈：
        </p>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>物理与算法维度</th><th>2D 透视视图 (Perspective View)</th><th>3D 鸟瞰视图 (Bird's-Eye View, BEV)</th><th>对自动驾驶核心系统的影响</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>目标尺度物理特性</b></td><td>随距离急剧衰减（5m 处轿车占 800px，50m 处仅占 40px，差距 20 倍）</td><td><b>绝对尺度恒定（一辆车在 BEV 网格中永远是长 4.5m × 宽 1.8m）</b></td><td>BEV 下卷积核与注意力的特征感受野具有物理确定性</td></tr>
                    <tr><td><b>目标空间遮挡关系</b></td><td>前景物体完全覆盖后景物体，造成像素级丢失</td><td><b>沿地表网格平铺展开，各物体在地面空间相互独立无重叠</b></td><td>大幅改善拥堵车流中被大货车遮挡的行人和跟车检测</td></tr>
                    <tr><td><b>度量物理单位</b></td><td>像素 (Pixel)，无直接物理距离概念</td><td><b>公制米 (Meter，如 0.4m 或 0.8m / 栅格)</b></td><td>直接输出可度量的欧氏距离，彻底消除像素到米的二次解算</td></tr>
                    <tr><td><b>下游规划控制对接</b></td><td>规控无法在透视像素上规划路径，需复杂几何反解</td><td><b>与高精地图 (HD Map) 和轨迹规划器完全共享后轴世界坐标系</b></td><td>天然消除感知与规控之间的信息孤岛，是端到端模型的理想底座</td></tr>
                </tbody>
            </table>
        </div>

        <!-- 数学几何公式推导：透视成像单向投影的尺度畸变 -->
        <div style="background: rgba(10, 14, 23, 0.9); padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border-subtle); margin: 1.5rem 0;">
            <h4 style="color: var(--accent-cyan); margin-bottom: 0.5rem;">📐 透视投影下的尺度坍缩物理数学公式</h4>
            <div style="background: #020408; padding: 0.85rem 1.1rem; border-radius: 6px; border: 1px solid rgba(0,229,255,0.2); margin-bottom: 0.75rem; overflow-x: auto;">
                $$s = \frac{f \cdot S_{\text{real}}}{Z_{\text{depth}}} \implies \frac{\partial s}{\partial Z} = -\frac{f \cdot S_{\text{real}}}{Z^2}$$
            </div>
            <div style="font-size: 0.86rem; color: var(--text-secondary); line-height: 1.7;">
                其中 $f$ 为像机物理焦距，$S_{\text{real}}$ 为物体实际物理尺寸，$Z$ 为障碍物距相机的纵向距离。
                由于像素尺寸 $s$ 与距离 $Z$ 成反比，当 $Z$ 较大时，像素对距离的导数急剧减小，
                导致<b>远距离物体的 1 个像素轻微抖动，会在反求 3D 物理空间时引起高达 3~5 米的剧烈测距震荡</b>！
            </div>
        </div>
    """ + how_to_read("为什么 2D 检测框无法直接做高精测距？", [
        ("地表起伏破坏先验", "许多单目测距算法依靠检测框底部接地线（Bottom Edge）假定其在平地上来推算距离。然而当路面出现坡道、减速带或车辆悬架起伏时，接地线高度偏差 5cm 会导致 50m 外纵向测距暴跌 10m 误差！"),
        ("视角分界线截断", "当目标车辆跨越两路相邻相机边界时，2D 检测头只能在每个相机中检出半截车身，由于长宽比被破坏，回归出的深度完全失真。")
    ]) + interview_card("面试官常问：既然 BEV 优势如此明显，为什么 2021 年之前学术界和工业界普遍不做多相机端到端 BEV，而是直到 LSS 和 BEVDet 之后才全面爆发？", 
                         "【答题思路与算力演进】：<br>"
                         "1. <b>跨视角空间变换的计算代价</b>：2D 卷积依赖局部像素近邻先验，而将 6 路图像投射到统一 BEV 网格，是一个全局、离散、非局部的几何重映射过程。早期的纯几何单应性（IPM）存在高度拉伸；而 3D 卷积在车载计算平台上显存完全吃不消。<br>"
                         "2. <b>算子基础设施不成熟</b>：在 BEV Pooling v2 出现前，基于累加和（cumsum）的 Splat 操作单帧耗时超过 50ms，无法在任何车端平台做到 10FPS 实时。<br>"
                         "3. <b>大规模多模态数据集驱动</b>：直到 nuScenes 和 Waymo 提供了高精度同步标注与外参真值，学术界才拥有了统一评测 3D NDS 指标的数据基石。")

    c = card("透视视图 (Perspective) 与鸟瞰视图 (BEV) 物理属性深度对比", "SCHEMATIC", "badge-schematic", body)
    return f'<div id="subpage-ch03-sub01" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch03_sub02():
    h = hero("SCHEMATIC", "badge-schematic", "Ego-Centric BEV Space Unification", 
             "3.2 鸟瞰视角（BEV, Bird's-Eye View）的物理优势", 
             "理解 BEV 空间表征的三大终极价值：多传感器自然无缝拼接、卷积尺度的物理平移不变性，以及基于自车位姿的时序对齐。")
    
    body = """
        <div class="guide-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
            <div class="guide-item" style="border-left-color: var(--accent-cyan); background: rgba(0, 229, 255, 0.04); padding: 1.25rem;">
                <div class="guide-item-title" style="color: var(--accent-cyan); font-size: 1rem;">1. 多相机自然无缝拼接 (Cross-Camera Mosaic)</div>
                <div class="guide-item-text" style="font-size: 0.88rem; line-height: 1.6;">
                    在 BEV 空间中，6 路相机的 2D 图像特征不再按图像边界割裂，而是根据精确几何内外参映射到同一个连续的俯视网格中。
                    位于两相机重叠区域的同一辆车，其投射特征在物理空间自然对齐聚合，彻底消除传统 2D 方案的“跨相机 ID 匹配”和“边界重复误检”难题。
                </div>
            </div>

            <div class="guide-item" style="border-left-color: var(--accent-purple); background: rgba(139, 92, 246, 0.04); padding: 1.25rem;">
                <div class="guide-item-title" style="color: var(--accent-purple); font-size: 1rem;">2. 尺度物理平移不变性 (Physical Scale Invariance)</div>
                <div class="guide-item-text" style="font-size: 0.88rem; line-height: 1.6;">
                    在 2D 图像上，网络必须使用多尺度特征金字塔 (FPN) 强行适应远小近大的剧烈尺度缩放。
                    而在米制 BEV 空间中，所有车辆的物理尺寸在 0.8m x 0.8m 栅格下占据固定的 5 x 2 像素，卷积核在全图滑动时感受野与物体真实物理尺寸完全吻合。
                </div>
            </div>

            <div class="guide-item" style="border-left-color: var(--accent-green); background: rgba(0, 230, 118, 0.04); padding: 1.25rem;">
                <div class="guide-item-title" style="color: var(--accent-green); font-size: 1rem;">3. 天然支持时序自车对齐 (Temporal Ego Alignment)</div>
                <div class="guide-item-text" style="font-size: 0.88rem; line-height: 1.6;">
                    当车辆在 3D 物理空间运动时，历史帧 BEV 特征与当前帧 BEV 特征的转换关系严格等价于一个 2D 刚体仿射旋转平移。
                    只需调用一次 PyTorch 的 <code>F.grid_sample</code>，即可在亚毫秒内将过去多个时间戳的特征对齐到当前坐标系，使多帧时序融合变得前所未有的轻量和优雅。
                </div>
            </div>
        </div>
    """ + gotcha_box("BEV 栅格分辨率与感知距离的立方算力陷阱", 
                    "许多初学者误以为 BEV 分辨率越高越好。设物理范围为 [-51.2m, 51.2m]。若栅格尺寸从 0.8m 细化为 0.2m，网格数量从 128x128=16,384 暴涨至 512x512=262,144（整整增长 16 倍！）。后续 BEV 骨干卷积与多头注意力的显存和计算量按平方暴增，导致推理帧率直接跌破实时门槛。工程落地中通常在感知距离、网格尺寸与主干算力之间做严格的 Pareto 权衡。")

    c = card("BEV 表示法的三大终极物理价值", "SCHEMATIC", "badge-schematic", body)
    return f'<div id="subpage-ch03-sub02" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch03_sub03():
    h = hero("SCHEMATIC", "badge-schematic", "Homography Flat-Ground vs Modern View Transformations", 
             "3.3 逆透视映射（IPM）vs 深度学习视图变换 (LSS 经典范式)", 
             "对比经典几何 IPM 与现代深度学习两大流派：自底向上（Bottom-Up: LSS/BEVDet）与自顶向下（Top-Down: BEVFormer/Sparse4D）。")
    
    ipm_link = term_link('ipm_concept', 'IPM')
    p_intro = f"""
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            传统计算机视觉中最直观的 2D 转 BEV 方法是<b>逆透视映射 (Inverse Perspective Mapping, {ipm_link})</b>。
            IPM 依赖极其强烈的几何假设——<b>平地假设（Flat-Ground Assumption）</b>：假定视野内所有像素对应的物理点都落在 $Z=0$ 的平面上。
            在此假设下，透视投影公式中的深度 $Z$ 被强制锁定为常数，相机与地面之间构成一个确定可逆的 $3 \\times 3$ 单应性矩阵 $H$。
        </p>
    """
    
    latex_body = r"""
        <!-- 论文主架构图与核心模块拆解 (Point 1 核心增加) -->
        <div class="media-container" style="margin-bottom: 1.5rem;">
            <img src="learning_report/images/paper_architectures/lss_architecture_diagram.png" alt="LSS Official Architecture Diagram" onclick="openLightbox(this.src, 'LSS (Lift, Splat, Shoot) 官方论文总体架构与张量变换管线', 'PAPER ARCHITECTURE', 'badge-real-data')">
            <div class="media-caption">
                📐 <b>LSS (Lift, Splat, Shoot - ECCV 2020) 论文主框架图</b>：6 视角相机输入 ➔ 2D FPN 上下文向量 $C$ 与离散深度分布 $P(d)$ ➔ 外积抬升为视锥点云 $C \otimes \alpha$ ➔ 标定外参几何投影 ➔ Voxel Pooling 栅格聚合 ➔ BEV 卷积与规划端到端输出。
            </div>
        </div>

        <!-- 和普通 2D CV 的本质区别深度剖析 -->
        <div style="background: rgba(14, 165, 233, 0.06); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 8px; padding: 1.1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #38BDF8; font-size: 1.02rem; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🔬 核心对比：LSS 视图变换与传统 2D CV 的三大本质断代区别</span>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.75;">
                <ul style="padding-left: 1.2rem; margin: 0;">
                    <li style="margin-bottom: 0.5rem;">
                        <b>1. 逆投影不适定性的概率化解 (Probabilistic Depth)</b>：传统 2D CV 中单张图片无法恢复 3D（一根光线穿过无穷个三维点，单像素到 3D 是病态逆问题）。LSS 并不做“硬性”深度点预测，而是输出离散深度分类概率分布 $\alpha \in \mathbb{R}^D$（例如 $D=118$ 个深度区间，从 4m 到 45m），沿光线将 2D 特征赋予概率加权，将不适定问题转化为概率期望累积。
                    </li>
                    <li style="margin-bottom: 0.5rem;">
                        <b>2. 显式几何光线投影 (Explicit Ray Casting)</b>：传统 2D 卷积只在 $(u,v)$ 邻域内做点乘，毫无三维物理约束。LSS 利用严格的针孔相机内参矩阵 $K$ 与外参 $[R|T]$，将每个视锥点直接反投影到自车物理坐标：
                        $$P_{\text{ego}} = R_{\text{cam}\to\text{ego}} \cdot \left( d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right) + T_{\text{cam}\to\text{ego}}$$
                        使得特征流动完全遵循刚体运动学和光学几何定律。
                    </li>
                    <li>
                        <b>3. 跨相机视场重叠自然融合 (Cross-Camera Voxel Pooling)</b>：传统 2D 检测在相邻相机重叠处会产生两个独立的目标检测框，需要复杂的 NMS 或跨相机后处理。而在 LSS 的 Splat 阶段，来自不同相机的光线点云落入同一个 BEV 柱体（Pillar）时，直接通过特征求和（Sum Pooling）完成融合，实现了天然无缝的全景拼接。
                    </li>
                </ul>
            </div>
        </div>

        <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <h4 style="color: #F87171; margin-bottom: 0.5rem; font-size: 0.95rem;">🚨 为什么 IPM 无法用于 3D 障碍物检测？</h4>
            <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6;">
                现实中的车辆、行人、路灯均具有立体垂直高度（$Z \\neq 0$）。
                当 IPM 强行将一辆 1.5 米高轿车的车顶像素按 $Z=0$ 反投影时，由于该像素在像平面中高度较高，单应矩阵会将其错误判定为“前方 60 米处的地面点”，
                从而在 BEV 俯视图上将整辆车拉伸成一条长达几十米的<b>“面条状辐射虚影（Noodle Streaking Artifact）”</b>，彻底摧毁 3D 边界框回归！
            </p>
        </div>

        <div class="how-to-read-box">
            <div class="how-to-read-title">🚀 现代深度学习解决视图变换的两大主干流派</div>
            <div class="guide-grid">
                <div class="guide-item" style="border-left-color: var(--accent-cyan);">
                    <div class="guide-item-title">流派一：自底向上 (Bottom-Up: LSS, BEVDet, BEVDepth)</div>
                    <div class="guide-item-text">
                        <b>“数据推导范式”</b>：从 2D 像素出发，由网络显式预测离散深度分布 $P(d|u,v)$，将 2D 像素特征乘以深度概率沿视锥光线“抬升（Lift）”展开为 3D 视锥点云，再“喷射（Splat）”体素求和池化到 BEV 网格中。
                        优势是几何解释性极强，推理速度快，适合硬件固化算子。
                    </div>
                </div>
                <div class="guide-item" style="border-left-color: var(--accent-purple);">
                    <div class="guide-item-title">流派二：自顶向下 (Top-Down: BEVFormer, PETR, Sparse4D)</div>
                    <div class="guide-item-text">
                        <b>“按需索取范式”</b>：在 BEV 网格或 3D 物理空间中预定义一组可学习的查询向量（BEV Query 或 3D Anchors），
                        利用交叉注意力机制（Cross-Attention）或坐标编码，让 3D 查询主动到 2D 多视角图像中“采样查询”所需特征。
                        优势是摆脱了脆弱的单像素单目绝对深度估计，对高度起伏容忍度极佳。
                    </div>
                </div>
            </div>
        </div>
    """
    body = p_intro + latex_body + interview_card("面试官常问：请从数学本质和工业部署角度，深入对比自底向上 (LSS) 与自顶向下 (BEVFormer) 的优劣势与适用场景？", 
                         "【答题全方位对比矩阵】：<br>"
                         "1. <b>深度估计鲁棒性</b>：LSS 高度依赖单目深度估计的准确性。在弱纹理区域或夜间，一旦 DepthNet 预测偏了，3D 特征会直接被‘甩’到错误位置；BEVFormer 采用 3D 立柱采样多点聚合，不强制单点硬性深度，抗深度扰动性更强。<br>"
                         "2. <b>算力与显存开销</b>：LSS 视锥张量展开后占用大量显存，但通过定制 CUDA 算子（BEV Pooling v2）可将 Splat 压缩到 1ms，在量产芯片（Orin / J5）上部署极其成熟；BEVFormer 计算复杂度直接受 Query 数量制约，高分辨率下可形变注意力访存开销大，量产量化部署难度相对较高。<br>"
                         "3. <b>选型结论</b>：注重极高帧率、低算力成本的量产项目优选 BEVDet/BEVDepth；注重长时序感知精度、大范围多任务融合的旗舰领航方案优选 BEVFormer 或 Sparse4D。")

    c = card("IPM 平地假设与深度学习方案演进", "SCHEMATIC", "badge-schematic", body)
    return f'<div id="subpage-ch03-sub03" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch03():
    return "\n".join([
        build_ch03_sub01(),
        build_ch03_sub02(),
        build_ch03_sub03()
    ])
