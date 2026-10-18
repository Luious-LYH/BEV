"""
Chapter 01: 数据闭环：nuScenes 真实多模态结构
Pedagogical and engineering deep-dive builder for subpages 1.1 to 1.5.
Compatible with Python 3.9+ (modular subpage functions, no nested f-strings).
"""

from scripts.builder.common import hero, card, how_to_read, interview_card, gotcha_box, tensor_flow_box, term_link

def build_ch01_sub01():
    h = hero("REAL DATA", "badge-real-data", "nuScenes Multi-Modal Relational Database", 
             "1.1 一帧自动驾驶数据到底是什么？", 
             "拆解 nuScenes 真实数据底座的关系型拓扑，厘清 scene、sample、sample_data、ego_pose 与标注真值之间的时空关联。")
    
    body = f"""
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            在自动驾驶工程中，“一帧数据”绝不仅仅是一张图片或一个点云文件，而是一个严格时空对齐的<b>高维多模态物理测量集合</b>。
            nuScenes 数据集采用类似关系型数据库的外键关联模式进行组织。以我们项目中真实运行的 <code>scene-0061</code>（波士顿雨天街景）为例，其核心拓扑链条如下：
        </p>

        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-family: monospace; font-size: 0.86rem; color: var(--accent-cyan); line-height: 1.8;">
                <b>Scene (20秒连续行驶片段, 唯一标识: scene-0061)</b><br>
                &nbsp;&nbsp;└── <b>Sample (2Hz 标注关键帧, 每隔约 0.5s 抽取一帧，Token: ca9a282c9e77460f...)</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <b>Sample_Data (各传感器独立原始数据流)</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── 📷 CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT (12Hz JPEG 流)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── 📷 CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT (12Hz JPEG 流)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── 📡 LIDAR_TOP (20Hz Velodyne 32线激光雷达 .pcd.bin 流)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── 📻 RADAR_FRONT, RADAR_FRONT_LEFT... (13Hz 毫米波雷达 .pcd 流)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <b>Ego_Pose (高频 IMU/底盘里程计自车位姿: 平移 translation [3] + 四元数 rotation [4])</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <b>Calibrated_Sensor (出厂标定内外参: 刚体外参 T_sensor2ego + 相机透视内参 K)</b><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <b>Sample_Annotation (360° 3D 目标真值边界框: [x,y,z,w,l,h,yaw], 类别, 速度属性)</b>
            </div>
        </div>
        """ + tensor_flow_box([
            ("Raw Disks", "6 JPEG (1600x900) + 1 Bin (34k pts)", "磁盘原始二进制文件，由专用 I/O 模块异步预读加载"),
            ("Batch Loader", "[B=1, N=6, 3, 900, 1600]", "经过解码、归一化后进入 PyTorch 张量，配合数据增强 (Crop/Resize/Flip)"),
            ("Extrinsics Matrix", "[B, 6, 4, 4] & [B, 1, 4, 4]", "刚体旋转平移齐次外参，将各传感器对齐至后轴自车系 (Ego Frame)"),
            ("Intrinsics K", "[B, 6, 3, 3]", "相机针孔模型内参，包含焦距 fx, fy 与主点 cx, cy")
        ]) + how_to_read("如何验证一帧多模态数据的质量？", [
            ("时间戳对齐窗口", "检查 6 路相机与激光雷达的时间戳偏差（Timestamp Jitter）。工业标准要求硬件同步触发误差在 5ms 以内，若超过 20ms，在 60km/h 车速下会造成 0.33 米以上的空间重投影错位。"),
            ("数据缺失与掉帧容错", "工程流水线必须具备 Drop-frame 防御逻辑。若右后相机临时丢帧，不可抛出 Exception 导致自动驾驶系统崩溃，需使用零张量或上一帧缓存补齐。"),
            ("外参标定有效性", "检查各相机的旋转四元数模长是否严格等于 1.0。若经过微调后破坏了单位正交性，矩阵逆变换将产生非物理的形变缩放。")
        ]) + gotcha_box("CAN 总线与视觉时间戳的时钟漂移 (Clock Drift)", 
                       "自车里程计位姿 (Ego Pose) 通常来源于车身 CAN 总线或高精度组合惯导系统，其系统时钟可能与传感器主机 (IPC) 存在微小的微秒级时钟漂移。如果直接用 sample 时间戳去最近邻匹配 ego_pose，在高速转弯时会导致累积航向角误差。工业标准做法是对前后两个最近位姿时间戳进行三次样条或四元数球面插值 (Slerp)。"
        ) + interview_card("面试官常问：nuScenes 数据集中 sample 与 sample_data 有何本质区别？为什么不把所有采集到的 12Hz/20Hz 图像全部人工打标？", 
                           "【答题要点与行业成本考量】：<br>"
                           "1. <b>数据层级区别</b>：<code>sample_data</code> 代表传感器底层无间断高频采集的原始物理数据流（相机 12Hz，激光 20Hz）；而 <code>sample</code> 则是从中以 2Hz（约 0.5 秒）周期抽取出来的“高价值关键帧”。<br>"
                           "2. <b>标注成本与信息冗余</b>：在 3D 空间进行高精度点云与环视多视角联合真值标注极其昂贵（每张 3D 关键帧全量标注成本高达数百元人民币）。相邻两帧（如 12Hz 相隔 83ms）车辆位移极小，全量打标存在 90% 以上的无用信息冗余。<br>"
                           "3. <b>时序学习优势</b>：未打标的中间 <code>sample_data</code> 绝非废弃，而是作为 " + term_link('keyframe_sweep_concept', 'Sweeps 子帧') + " 供时序网络（如 BEVStereo、SparseOcc）构建多帧特征累积与双目几何代价体，以零标注成本换取巨大的时序感知增益。"
        ) + """
        <div class="action-btn-row">
            <button class="btn-secondary" onclick="openTensorModal('raw_lidar')">📊 查看原始点云张量元数据</button>
            <button class="btn-secondary" onclick="openCodeModal('lidar_parsing')">💻 查看底层数据解析代码</button>
        </div>
        """

    c = card("nuScenes 核心数据架构关系图谱与拓扑分解", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch01-sub01" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch01_sub02():
    h = hero("REAL DATA", "badge-real-data", "6 Surround-View Cameras Rig", 
             "1.2 六路环视相机（Surround Cameras）物理布局与画面拼图", 
             "在可交互 3D 视锥空间中透视 6 路车规级摄像头的安装外参、视场角 (FOV) 空间重叠与拼接盲区。")
    
    body1 = """
        <div class="media-container">
            <img src="learning_report/images/real/nuscenes/01_surround_cameras.png" alt="6 Surround View Cameras" onclick="openLightbox(this.src, '真实 nuScenes 6 路环视相机同步拼图', 'REAL DATA', 'badge-real-data')">
            <div class="media-caption">
                📷 真实 nuScenes 环视拼图：上方为前向三路（左前 CAM_FRONT_LEFT、前视 CAM_FRONT、右前 CAM_FRONT_RIGHT），下方为后向三路（左后 CAM_BACK_LEFT、后视 CAM_BACK、右后 CAM_BACK_RIGHT）。分辨率 1600x900。
            </div>
        </div>
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            从 6 路真实拼图中可以清晰观察到多相机系统的物理特性：
            各相机由于朝向和光照差异，其白平衡、自动曝光与反光表现截然不同；
            相邻相机之间存在大约 <b>10%~15% 的水平视场重叠区域 (Overlap Region)</b>。
            例如在 <code>CAM_FRONT_LEFT</code> 的右侧边缘与 <code>CAM_FRONT</code> 的左侧边缘，同一栋建筑物的立柱在两幅透视图像中同时出现，但呈现完全相反的透视斜角！
        </p>
    """ + how_to_read("如何从多相机拼图中解读几何拼接难点？", [
        ("重叠区域的同一目标", "若采用传统 2D 单目检测后融合方案，同一辆车会同时在两张图上被检出，在做后处理 NMS 拼接时极易因深度误差产生两个重叠的 3D 伪目标。"),
        ("车身近景遮挡盲区", "由于相机安装在车顶四周，车辆正下方贴地区域（地面盲区）在各相机视锥之外，纯视觉必须依靠车位地锁历史记忆或超声波/环视鱼眼 AVM 进行盲区补充。"),
        ("焦距与视野权衡", "CAM_FRONT 采用长焦以看清 100m 外的交通灯与前车；侧后相机采用较广视野以捕获变道盲区 (BSD)。")
    ])

    c1 = card("6 路环视相机同步拼图 (Surround View Mosaic)", "REAL DATA", "badge-real-data", body1)

    body2 = """
        <p style="color: var(--text-secondary); margin-bottom: 0.75rem; font-size: 0.9rem;">
            💡 <b>3D 控制指南：</b>支持 3D 自由旋转（鼠标左键拖拽）、缩放（滚轮）、平移（Shift+拖拽）。点击预设视角快速切换，点击下方传感器胶囊按钮高亮查看真实安装参数与视锥锥体。
        </p>
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap;">
            <button class="btn-secondary" onclick="setRigPreset('3d')">📐 3D 自由透视</button>
            <button class="btn-secondary" onclick="setRigPreset('top')">🛸 BEV 正俯视 (Z)</button>
            <button class="btn-secondary" onclick="setRigPreset('front')">🚘 前视正视 (Front)</button>
            <button class="btn-secondary" onclick="setRigPreset('side')">🚙 侧视 (Side)</button>
        </div>
        <div style="display: flex; gap: 0.35rem; margin-bottom: 1rem; flex-wrap: wrap;" id="rig-sensor-pill-container">
            <button class="btn-secondary active rig-sensor-pill" data-sensor="CAM_FRONT" onclick="selectRigSensor('CAM_FRONT')">📷 CAM_FRONT</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="CAM_FRONT_LEFT" onclick="selectRigSensor('CAM_FRONT_LEFT')">📷 CAM_FRONT_LEFT</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="CAM_FRONT_RIGHT" onclick="selectRigSensor('CAM_FRONT_RIGHT')">📷 CAM_FRONT_RIGHT</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="CAM_BACK" onclick="selectRigSensor('CAM_BACK')">📷 CAM_BACK</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="CAM_BACK_LEFT" onclick="selectRigSensor('CAM_BACK_LEFT')">📷 CAM_BACK_LEFT</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="CAM_BACK_RIGHT" onclick="selectRigSensor('CAM_BACK_RIGHT')">📷 CAM_BACK_RIGHT</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="LIDAR_TOP" onclick="selectRigSensor('LIDAR_TOP')" style="color: #00E676;">📡 LIDAR_TOP</button>
            <button class="btn-secondary rig-sensor-pill" data-sensor="RADAR_FRONT" onclick="selectRigSensor('RADAR_FRONT')" style="color: #FFAB00;">📻 RADAR_FRONT</button>
        </div>
        <div class="viewer-3d-box" style="height: 480px; position: relative;">
            <canvas id="rig-canvas-3d" style="width: 100%; height: 100%;"></canvas>
        </div>
        <div id="rig-info-card" class="how-to-read-box" style="margin-top: 1rem;"></div>
    """ + interview_card("面试官常问：自动驾驶的自车坐标系 (Ego Coordinate) 为什么通常定在后轴中心在地面投影，而不是车身几何中心或前轴？", 
                         "【答题核心机理与车辆运动学】：<br>"
                         "1. <b>后轴中心是车辆运动学的瞬时旋转极点</b>：根据阿克曼转向几何学与自行车运动学模型 (Kinematic Bicycle Model)，前轮具有转向转角 $\\delta$，而普通乘用车的后轮只有纯滚动无横摆转向。<br>"
                         "2. <b>方程无奇点简化</b>：以车辆后轴中心为原点建立运动学方程时，车体在瞬时切线方向的速度矢量严格与车身纵轴 Y 对齐（侧向滑移角 $\\beta \\approx 0$），运动学转移矩阵没有由前轮横摆角耦合引起的复杂三角函数高阶非线性项。<br>"
                         "3. <b>轨迹规划统一</b>：控制底盘的阿克曼底盘执行器是以驱动后轴为控制基准点的，感知输出若直接位于后轴中心，规划控制无需再做二次底盘力学偏置转换。")

    c2 = card("🚗 交互式车辆传感器物理安装拓扑 (Sensor Rig 3D Viewer)", "INTERACTIVE 3D", "badge-interactive", body2)
    return f'<div id="subpage-ch01-sub02" class="subpage-content-section" style="display:none;">{h}\n{c1}\n{c2}</div>'

def build_ch01_sub03():
    h = hero("REAL DATA", "badge-real-data", "Velodyne HDL-32E 32-Line Point Cloud", 
             "1.3 LiDAR 原始二进制解析与点云特性", 
             "从底层 IEEE-754 单精度浮点二进制流剖析 34,688 个激光点，掌握点云空间三维坐标、反射强度与线束分布。")
    
    body = """
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            激光雷达采集的数据通常以紧凑的二进制格式落盘以节省 I/O 带宽。nuScenes 使用 <code>.pcd.bin</code> 格式，
            每个点由 5 个连续的 32 位小端浮点数（float32，占 20 字节）编码：
            <code>[x, y, z, intensity, ring_index]</code>。
            一帧包含 34,688 个点，文件大小为 <code>34,688 × 20 = 693,760 字节 ≈ 677.5 KB</code>。
        </p>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>点序号</th><th>X (米, 左右)</th><th>Y (米, 前后)</th><th>Z (米, 高低)</th><th>反射强度 (Intensity)</th><th>线束号 (Ring)</th><th>物理目标解译</th></tr>
                </thead>
                <tbody>
                    <tr><td>#0</td><td>4.125</td><td>1.214</td><td>-0.452</td><td>18.0</td><td>3</td><td>自车右侧 4.3m 处沥青地面漫反射</td></tr>
                    <tr><td>#1</td><td>4.180</td><td>1.230</td><td>-0.458</td><td>19.0</td><td>3</td><td>相邻发射光束地面回波点</td></tr>
                    <tr><td>#5</td><td>7.821</td><td>-2.451</td><td>-0.320</td><td>38.0</td><td>12</td><td>右侧金属护栏强反射点（金属介电常数高）</td></tr>
                    <tr><td>#24</td><td>-3.142</td><td>12.850</td><td>0.450</td><td>85.0</td><td>21</td><td>前向 13m 处前车车牌与逆反射反光条（极高能量）</td></tr>
                    <tr><td>#88</td><td>26.540</td><td>3.410</td><td>0.880</td><td>5.0</td><td>27</td><td>26m 外远景树叶弱散射（能量急剧衰减）</td></tr>
                </tbody>
            </table>
        </div>

        <h4 style="color: var(--accent-cyan); margin: 1.5rem 0 0.5rem;">🌌 真实激光雷达点云 3D 交互视检器 (LIDAR_TOP 34,688 Points)</h4>
        <div id="lidar-metadata-banner" style="font-family: monospace; font-size: 0.8rem; color: var(--accent-green); margin-bottom: 0.5rem;"></div>
        
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
            <button class="btn-secondary active" onclick="if(lidarViewerInstance)lidarViewerInstance.setColorMode('height')">🌈 按高度着色 (Z Height)</button>
            <button class="btn-secondary" onclick="if(lidarViewerInstance)lidarViewerInstance.setColorMode('intensity')">⚡ 按反射强度着色 (Intensity)</button>
            <button class="btn-secondary" onclick="if(lidarViewerInstance)lidarViewerInstance.setColorMode('ring')">🎯 按线束编号着色 (Ring)</button>
        </div>

        <div class="viewer-3d-box" style="height: 500px;">
            <canvas id="lidar-canvas-3d" style="width: 100%; height: 100%;"></canvas>
        </div>
        <div id="lidar-stats" style="font-family: monospace; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;"></div>
    """ + gotcha_box("点云‘近密远稀’二次方衰减物理定律", 
                    "激光雷达发射角度分辨率固定（如水平 0.2°，垂直 1.33°）。随着距离增加，相邻两根光束之间的物理空间弧长按 $R \\cdot \\Delta \\theta$ 线性扩大，单位表面积上的激光落点密度按 $1/R^2$ 剧烈衰减！在 5 米处一辆小轿车能命中上千个点，但在 50 米处仅能命中 3~5 个点。纯激光雷达算法在远距离极易漏检，这正是多相机 BEV 融合的刚需价值所在。"
    ) + interview_card("面试官常问：激光点云的强度 (Intensity) 通道物理含义是什么？在感知算法中有什么关键作用？", 
                       "【答题深度要点】：<br>"
                       "1. <b>物理本质</b>：强度通道度量激光雷达探测器接收到的回波光脉冲能量幅值。它受目标物体表面的材料反射率 (Reflectance)、入射角以及传播距离衰减三者共同决定。<br>"
                       "2. <b>车道线与路面标识检测</b>：沥青路面反射率通常很低 (Intensity 10~25)，而车道标线、斑马线、停止线喷涂了含玻璃微珠的逆反射涂料，反射率极高 (Intensity > 80)。仅通过强度阈值分割或特征卷积，即可在无相机输入时完成夜间车道线抽取。<br>"
                       "3. <b>高反虚警识别</b>：交通指示牌、路锥和前车反光条具有极高强度，常导致探测器饱和产生‘光晕扩张’伪点，强度通道是识别高反目标和排除噪点的必要特征。")

    c = card("LiDAR 二进制数组与真实点数据解剖", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch01-sub03" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch01_sub04():
    h = hero("REAL DATA", "badge-real-data", "5 Continental ARS-408 Radar Sensors", 
             "1.4 5 路毫米波雷达（Radar）的多普勒测量", 
             "理解毫米波雷达调频连续波 (FMCW) 多普勒效应，掌握瞬时径向相对速度与雷达散射截面积 (RCS)。")
    
    body = """
        <div class="media-container">
            <img src="learning_report/images/real/nuscenes/03_radar_bev.png" alt="Radar BEV Overlay" onclick="openLightbox(this.src, '5 路毫米波雷达在 BEV 俯视平面的速度矢量叠加', 'REAL DATA', 'badge-real-data')">
            <div class="media-caption">
                📻 5 路毫米波雷达在 BEV 俯视平面叠加：红色箭头为目标瞬时多普勒速度矢量。
            </div>
        </div>

        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            自动驾驶车辆在车身四周配备 5 颗毫米波雷达：1 颗正前向长距雷达 (RADAR_FRONT，探测距离达 250m)，
            以及 4 颗安装在四个保险杠拐角的中短距角雷达 (RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT, RADAR_BACK_LEFT, RADAR_BACK_RIGHT)。
            与激光雷达和相机相比，毫米波雷达最不可替代的物理特性是<b>全天候鲁棒性</b>（77GHz 毫米波波长约 4mm，能穿透雨雾扬尘）与<b>瞬时测速能力</b>。
        </p>

        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>传感器类型</th><th>单帧数据量</th><th>物理测距精度</th><th>瞬时测速能力</th><th>全天候抗恶劣天气能力</th><th>核心劣势</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>环视相机 (Camera)</b></td><td>极大 (6x 1600x900)</td><td>中等 (受限于深度估计)</td><td>无 (需跨帧光流/时序差分)</td><td>较弱 (受强逆光、大雨大雾影响)</td><td>深度存在不适定性</td></tr>
                    <tr><td><b>激光雷达 (LiDAR)</b></td><td>较大 (~3.5万点)</td><td>极高 (厘米级)</td><td>无 (需跨帧 ICP 匹配)</td><td>中等 (易受暴雪、水雾散射)</td><td>高精硬件成本极高</td></tr>
                    <tr><td><b>毫米波雷达 (Radar)</b></td><td>极小 (~100点)</td><td>高 (距离与径向准确)</td><td><b>有 (单脉冲多普勒频移直接测得)</b></td><td><b>极强 (穿透雨雪雾尘霾)</b></td><td>点极稀疏、缺乏垂直高程、虚警多</td></tr>
                </tbody>
            </table>
        </div>
    """ + interview_card("面试官常问：车载毫米波雷达只能测量径向速度，算法是如何求解出目标在自车坐标系下的完整 2D 绝对速度矢量 (vx, vy) 的？", 
                         "【答题思路与工程落地】：<br>"
                         "1. <b>多角雷达重叠视场几何联立解算</b>：当目标位于车身拐角同时被前向雷达和角雷达探测到时，设两条视线夹角为 $\\theta_1, \\theta_2$。联立两个径向投影方程 $v_{r1} = v_x \\cos\\theta_1 + v_y \\sin\\theta_1$ 与 $v_{r2} = v_x \\cos\\theta_2 + v_y \\sin\\theta_2$，直接解出唯一确定的 $(v_x, v_y)$。<br>"
                         "2. <b>单雷达时序卡尔曼滤波 (EKF)</b>：若只有单雷达覆盖，径向速度作为强观测，切向速度则通过前后帧的位置位移差分 $\\Delta y / \\Delta t$ 作为弱观测，在卡尔曼状态转移中完成协方差收敛。<br>"
                         "3. <b>端到端 Transformer 交叉注意力融合</b>：现代多模态模型（如 RC-BEV、Radar-BEV）将雷达点作为 3D Token 输入交叉注意力层，与环视图像特征联合优化，利用图像外观确定横向位移，雷达直接校准纵向速度。")

    c = card("毫米波雷达多普勒速度解算与全车布置", "REAL DATA", "badge-real-data", body)
    return f'<div id="subpage-ch01-sub04" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch01_sub05():
    h = hero("REAL DATA", "badge-real-data", "2Hz Keyframes vs 12Hz/20Hz Sweeps Temporal Alignment", 
             "1.5 Keyframe（关键帧）与 Sweeps（时序子帧）", 
             "彻底搞清自动驾驶数据闭环的时序结构：2Hz 标注关键帧与高频 Sweeps 子帧的拓扑关系、自车位姿累积与多帧点云稠密化。")
    
    svg_html = """
        <div style="background: #060913; border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 8px; padding: 1.5rem 1rem; margin-bottom: 1.5rem; overflow-x: auto;">
            <svg viewBox="0 0 950 320" style="width: 100%; min-width: 800px; height: auto;" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="kfGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#00E5FF"/>
                        <stop offset="100%" stop-color="#0072FF"/>
                    </linearGradient>
                    <linearGradient id="sweepGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#8B5CF6"/>
                        <stop offset="100%" stop-color="#D946EF"/>
                    </linearGradient>
                    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>

                <line x1="80" y1="70" x2="880" y2="70" stroke="rgba(0, 229, 255, 0.3)" stroke-width="3" stroke-dasharray="6,4"/>
                <line x1="80" y1="210" x2="880" y2="210" stroke="rgba(139, 92, 246, 0.3)" stroke-width="2"/>

                <text x="80" y="30" fill="#64748B" font-size="12" font-family="monospace">t = 0.0s</text>
                <text x="480" y="30" fill="#64748B" font-size="12" font-family="monospace">t = 0.5s (+500ms)</text>
                <text x="880" y="30" fill="#64748B" font-size="12" font-family="monospace">t = 1.0s (+1000ms)</text>

                <g transform="translate(80, 70)">
                    <circle cx="0" cy="0" r="28" fill="url(#kfGrad)" filter="url(#glow)"/>
                    <text x="0" y="5" fill="#000" font-size="13" font-weight="bold" text-anchor="middle" font-family="monospace">KF 0</text>
                    <rect x="-65" y="36" width="130" height="42" rx="4" fill="#0B132B" stroke="#00E5FF" stroke-width="1"/>
                    <text x="0" y="52" fill="#00E5FF" font-size="10" font-weight="bold" text-anchor="middle">KEYFRAME t-1</text>
                    <text x="0" y="68" fill="#94A3B8" font-size="9" text-anchor="middle">带 3D Box 全量真值</text>
                </g>

                <g transform="translate(480, 70)">
                    <circle cx="0" cy="0" r="32" fill="#00E5FF" filter="url(#glow)"/>
                    <circle cx="0" cy="0" r="25" fill="#030712"/>
                    <text x="0" y="5" fill="#00E5FF" font-size="13" font-weight="bold" text-anchor="middle" font-family="monospace">KF 1</text>
                    <rect x="-75" y="42" width="150" height="46" rx="4" fill="#0B132B" stroke="#00E5FF" stroke-width="1.5"/>
                    <text x="0" y="58" fill="#00E5FF" font-size="10" font-weight="bold" text-anchor="middle">KEYFRAME t (当前帧)</text>
                    <text x="0" y="74" fill="#00E676" font-size="9" text-anchor="middle">★ 本项目核心评估帧</text>
                </g>

                <g transform="translate(880, 70)">
                    <circle cx="0" cy="0" r="28" fill="url(#kfGrad)" filter="url(#glow)"/>
                    <text x="0" y="5" fill="#000" font-size="13" font-weight="bold" text-anchor="middle" font-family="monospace">KF 2</text>
                    <rect x="-65" y="36" width="130" height="42" rx="4" fill="#0B132B" stroke="#00E5FF" stroke-width="1"/>
                    <text x="0" y="52" fill="#00E5FF" font-size="10" font-weight="bold" text-anchor="middle">KEYFRAME t+1</text>
                    <text x="0" y="68" fill="#94A3B8" font-size="9" text-anchor="middle">时序未来滑动窗口</text>
                </g>

                <path d="M 80 70 Q 280 110 480 70 Q 680 30 880 70" fill="none" stroke="#FFAB00" stroke-width="2" stroke-dasharray="4,3"/>
                <text x="280" y="105" fill="#FFAB00" font-size="10" font-family="monospace" text-anchor="middle">自车运动补偿 T_prev➔curr</text>

                <g transform="translate(140, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw1</text></g>
                <g transform="translate(190, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw2</text></g>
                <g transform="translate(240, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw3</text></g>
                <g transform="translate(290, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw4</text></g>
                <g transform="translate(340, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw5</text></g>
                <g transform="translate(390, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw6</text></g>
                <g transform="translate(440, 210)"><circle cx="0" cy="0" r="8" fill="#8B5CF6"/><text x="0" y="20" fill="#94A3B8" font-size="8" text-anchor="middle">Sw7</text></g>

                <path d="M 140 200 C 140 150, 470 120, 470 105" fill="none" stroke="rgba(139, 92, 246, 0.4)" stroke-width="1.2" stroke-dasharray="3,3"/>
                <path d="M 290 200 C 290 150, 475 120, 475 105" fill="none" stroke="rgba(139, 92, 246, 0.6)" stroke-width="1.2" stroke-dasharray="3,3"/>
                <path d="M 440 200 C 440 160, 480 130, 480 105" fill="none" stroke="rgba(139, 92, 246, 0.9)" stroke-width="1.5"/>

                <text x="280" y="165" fill="#D946EF" font-size="10" font-weight="bold" text-anchor="middle">多 Sweeps 点云叠加稠密化 (Multi-Sweep Accumulation)</text>
                <text x="280" y="180" fill="#94A3B8" font-size="9" text-anchor="middle">3.4万点 ➔ 30万点 (10 帧融合)</text>

                <text x="80" y="280" fill="#38BDF8" font-size="11" font-weight="bold">● 2Hz Keyframe: 标注真值帧 (无拖影基准)</text>
                <text x="440" y="280" fill="#C084FC" font-size="11" font-weight="bold">● 20Hz LiDAR Sweeps: 稠密点云源</text>
                <text x="750" y="280" fill="#FBBF24" font-size="11" font-weight="bold">● 12Hz Camera Sweeps: 立体视差源</text>
            </svg>
        </div>
    """

    body = """
        <p style="color: var(--text-primary); line-height: 1.7; margin-bottom: 1rem;">
            传统课件常采用白底低对比度图片展示时序，看不清时间戳与位姿关联。
            下方为专门针对暗色工业 IDE 研发的<b>原生 SVG 交互式时序拓扑图</b>：
            顶层为每隔 0.5 秒 (2Hz) 抽取一次的 <b>Keyframe（带 360° 3D 边界框全量人工标注）</b>；
            底层为 12Hz 相机与 20Hz 激光雷达无间断采集的 <b>Sweeps（无人工标注子帧）</b>。
        </p>
    """ + svg_html + tensor_flow_box([
        ("Single Keyframe PointCloud", "[N=34688, 5]", "单个 50ms 周期 Velodyne 32 线扫描，远距离极度稀疏，车身仅有 2~3 个点"),
        ("10-Sweeps Accumulated", "[N=312450, 5]", "利用自车位姿 T_sweep2curr 对齐过去 10 帧后，点云密度激增近 10 倍"),
        ("Relative Timestamp Channel", "[N, 5] -> dt in [0, 0.5s]", "每个点附带与当前关键帧的时间差 dt，供 3D 检测主干网络推算目标速度")
    ]) + interview_card("面试官常问：将过去 10 个 Sweeps 点云叠加到当前帧，虽然点云变密了，但高速行驶的动态车辆会出现‘严重拖影长尾’，算法如何解决这一矛盾？", 
                         "【答题深度与实战方案】：<br>"
                         "1. <b>问题物理根源</b>：自车位姿补偿（Ego-Motion Compensation）只能严格对齐静止背景（道路、树木、路灯）。动态目标（如 60km/h 相对运动的小汽车）在 0.5s 周期内会位移 8.3 米，导致 10 帧扫描出的车身像一条长长的毛毛虫拉伸开来。<br>"
                         "2. <b>解法 A（时序时间戳特征嵌入）</b>：在将点输入 PointPillars 或 VoxelNet 时，绝不能丢弃第 5 维相对时间戳 $dt = t_{\\text{curr}} - t_{\\text{sweep}}$。点云网络会将连续分布的 $dt$ 解析为速度梯度，直接回归出瞬时线速度矢量 $(v_x, v_y)$。<br>"
                         "3. <b>解法 B（先检测后时序补偿）</b>：在高端激光雷达感知大模型中，第一阶段先使用粗检测获取所有动态物体的 3D 边界框与估计速度，第二阶段对框内的点云施加反向目标速度补偿 $P_{\\text{comp}} = P - V_{\\text{obj}} \\cdot dt$，将毛毛虫点云强制压缩收缩回当前瞬时长方体表面！")

    c = card("Keyframe 与 Sweeps 现代暗色交互时序拓扑架构", "NATIVE DARK SVG", "badge-interactive", body)
    return f'<div id="subpage-ch01-sub05" class="subpage-content-section" style="display:none;">{h}\n{c}</div>'

def build_ch01():
    return "\n".join([
        build_ch01_sub01(),
        build_ch01_sub02(),
        build_ch01_sub03(),
        build_ch01_sub04(),
        build_ch01_sub05()
    ])
