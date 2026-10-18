"""
Common HTML component builders for BEV / Occupancy Perception Textbook.
Provides reusable, styled building blocks for academic & industry-grade curriculum pages.
"""

def hero(badge_text, badge_class, subtitle, title, objective):
    return f"""
    <div class="page-hero">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span class="badge {badge_class}">{badge_text}</span>
            <span style="font-size: 0.82rem; color: var(--text-muted); font-family: monospace;">{subtitle}</span>
        </div>
        <h1 class="page-hero-title">{title}</h1>
        <div class="page-objective-box">
            🎯 <b>本节核心目标：</b>{objective}
        </div>
    </div>
    """

def card(title, badge_text, badge_class, body_html):
    badge_html = f'<span class="badge {badge_class}">{badge_text}</span>' if badge_text else ''
    return f"""
    <div class="pedagogy-card">
        <div class="card-header-bar">
            <h3 class="card-title">{title}</h3>
            {badge_html}
        </div>
        {body_html}
    </div>
    """

def how_to_read(title, items):
    """
    items: list of (guide_title, guide_text)
    """
    items_html = ""
    for g_title, g_text in items:
        items_html += f"""
        <div class="guide-item">
            <div class="guide-item-title">{g_title}</div>
            <div class="guide-item-text">{g_text}</div>
        </div>
        """
    return f"""
    <div class="how-to-read-box">
        <div class="how-to-read-title">🔍 {title}</div>
        <div class="guide-grid">
            {items_html}
        </div>
    </div>
    """

def interview_card(question, answer):
    return f"""
    <div style="margin: 1.5rem 0 1rem; background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1.25rem 1.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem;">
            <span style="font-size: 1.1rem;">🎯</span>
            <span style="font-weight: 700; color: #FDE68A; font-size: 0.95rem;">大厂感知算法工程师必考高频面试题</span>
            <span class="badge" style="background: rgba(245,158,11,0.2); color: #F59E0B; border: 1px solid #F59E0B; font-size: 0.7rem; margin-left: auto;">INTERVIEW FOCUS</span>
        </div>
        <div style="font-weight: 600; color: #FFFFFF; font-size: 0.92rem; margin-bottom: 0.75rem; line-height: 1.5;">
            ❓ {question}
        </div>
        <div style="font-size: 0.88rem; color: var(--text-primary); line-height: 1.7; background: rgba(0,0,0,0.4); padding: 1rem 1.25rem; border-radius: 6px; border-left: 3px solid #F59E0B;">
            {answer}
        </div>
    </div>
    """

def gotcha_box(title, text):
    return f"""
    <div style="margin: 1.25rem 0; background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.25); border-left: 4px solid #EF4444; border-radius: 6px; padding: 1rem 1.25rem;">
        <div style="font-weight: 700; color: #FCA5A5; font-size: 0.9rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
            <span>⚠️</span> <span>工程踩坑指南 (Production Pitfall)：{title}</span>
        </div>
        <div style="font-size: 0.86rem; color: #FECACA; line-height: 1.6;">
            {text}
        </div>
    </div>
    """

def tensor_flow_box(steps):
    """
    steps: list of (stage_name, tensor_shape, note)
    """
    rows = ""
    for name, shape, note in steps:
        rows += f"""
        <tr>
            <td style="font-weight: 600; color: var(--accent-cyan);">{name}</td>
            <td style="font-family: monospace; color: var(--accent-green);">{shape}</td>
            <td style="font-size: 0.82rem; color: var(--text-secondary);">{note}</td>
        </tr>
        """
    return f"""
    <div style="margin: 1.25rem 0;">
        <div style="font-size: 0.88rem; font-weight: 600; color: var(--accent-green); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
            <span>🔄</span> 核心特征张量流转矩阵 (Tensor Flow Blueprint)
        </div>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr><th>计算阶段</th><th>张量形状 (Tensor Shape)</th><th>物理与语义含义</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>
    """

def term_link(concept_key, display_text):
    return f'<span class="term-link" data-concept="{concept_key}">{display_text}</span>'

def phenomenon_card(look_where, normal_pattern, warning_signals, engineering_insight, common_pitfall, interview_30s):
    return f"""
    <div class="phenomenon-card" style="margin: 1.5rem 0; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 8px; overflow: hidden;">
        <div style="background: rgba(0, 229, 255, 0.08); padding: 0.75rem 1.25rem; border-bottom: 1px solid rgba(0, 229, 255, 0.2); display: flex; align-items: center; justify-content: space-between;">
            <div style="font-weight: 700; color: #00E5FF; font-size: 0.95rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🔬</span> <span>感知诊断与现象解释卡 (Perception Diagnostic Blueprint)</span>
            </div>
            <span class="badge" style="background: rgba(0, 229, 255, 0.15); color: #00E5FF; border: 1px solid #00E5FF; font-size: 0.7rem;">OBSERVE ➔ DIAGNOSE ➔ INTERVIEW</span>
        </div>
        <div style="padding: 1.25rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #38BDF8;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; margin-bottom: 0.35rem;">👀 先看哪里 (Where to Look First)</div>
                <div style="font-size: 0.84rem; color: #E2E8F0; line-height: 1.6;">{look_where}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #34D399;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #34D399; margin-bottom: 0.35rem;">✅ 正常现象 (Normal Behavior)</div>
                <div style="font-size: 0.84rem; color: #E2E8F0; line-height: 1.6;">{normal_pattern}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #F87171;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #F87171; margin-bottom: 0.35rem;">⚠️ 异常警惕 (Warning Signals)</div>
                <div style="font-size: 0.84rem; color: #FECACA; line-height: 1.6;">{warning_signals}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #A78BFA;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #A78BFA; margin-bottom: 0.35rem;">💡 工程判断 (Engineering Diagnosis)</div>
                <div style="font-size: 0.84rem; color: #E2E8F0; line-height: 1.6;">{engineering_insight}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #FBBF24;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #FBBF24; margin-bottom: 0.35rem;">❌ 高频误区 (Common Misconception)</div>
                <div style="font-size: 0.84rem; color: #FEF3C7; line-height: 1.6;">{common_pitfall}</div>
            </div>
            <div style="background: rgba(0, 0, 0, 0.35); padding: 0.85rem 1rem; border-radius: 6px; border-left: 3px solid #F59E0B;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #F59E0B; margin-bottom: 0.35rem;">🎯 面试30秒答法 (Interview 30s Key)</div>
                <div style="font-size: 0.84rem; color: #FDE68A; line-height: 1.6;">{interview_30s}</div>
            </div>
        </div>
    </div>
    """

def video_player_card(video_src, webp_src, title, badge_text, badge_class, caption, metadata_dict, phenomenon=None):
    meta_tags = ""
    for k, v in metadata_dict.items():
        meta_tags += f'<span style="background: rgba(255,255,255,0.06); padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); font-family: monospace; font-size: 0.78rem;"><b style="color: var(--accent-cyan);">{k}:</b> {v}</span> '

    phenomenon_html = phenomenon if phenomenon else ""

    # Map to frame player parameters
    if "temporal_ego_alignment" in video_src:
        player_id = "player-ego-align"
        folder = "ego_alignment"
        total_frames = 17
    elif "bev_tracking" in video_src:
        player_id = "player-bev-track"
        folder = "bev_tracking"
        total_frames = 25
    elif "occupancy" in video_src:
        player_id = "player-occ-flow"
        folder = "occ_flow"
        total_frames = 20
    else:
        player_id = "player-generic"
        folder = "occ_flow"
        total_frames = 20

    first_frame = f"learning_report/videos/frames/{folder}/frame_000.jpg"

    return f"""
    <div class="pedagogy-card" style="border: 1px solid rgba(0, 229, 255, 0.3); background: rgba(10, 16, 28, 0.9);">
        <div class="card-header-bar" style="border-bottom: 1px solid rgba(0, 229, 255, 0.2); padding-bottom: 0.75rem; margin-bottom: 1rem;">
            <div>
                <h3 class="card-title" style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>🎬</span> <span>{title}</span>
                </h3>
                <div style="margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem;">
                    {meta_tags}
                </div>
            </div>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>

        <!-- 交互式动效视频播放器控制器 -->
        <div style="background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 8px 8px 0 0; padding: 0.65rem 1rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;">
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <button id="{player_id}-play-btn" class="btn-primary" style="padding: 0.35rem 0.75rem; font-size: 0.82rem;" onclick="togglePlay('{player_id}')">⏸ 暂停</button>
                <button class="btn-secondary" style="padding: 0.35rem 0.6rem; font-size: 0.82rem;" onclick="stepFrame('{player_id}', -1)">⏮ 上一帧</button>
                <button class="btn-secondary" style="padding: 0.35rem 0.6rem; font-size: 0.82rem;" onclick="stepFrame('{player_id}', 1)">⏭ 下一帧</button>
                <span id="{player_id}-counter" style="font-family: monospace; font-size: 0.82rem; color: var(--accent-cyan); margin-left: 0.5rem;">帧 1 / {total_frames} (0.0s)</span>
            </div>

            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <label style="font-size: 0.78rem; color: var(--text-muted); display: flex; align-items: center; gap: 0.3rem;">
                    <span>⚡ 播放速度:</span>
                    <select style="background: #020408; color: #fff; border: 1px solid var(--border-subtle); padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.78rem;" onchange="setPlayerSpeed('{player_id}', this.value)">
                        <option value="2">0.5x 慢放</option>
                        <option value="4" selected>1.0x 正常</option>
                        <option value="8">2.0x 快速</option>
                    </select>
                </label>
                <button id="{player_id}-mode-btn" class="btn-secondary" style="padding: 0.25rem 0.55rem; font-size: 0.76rem;" onclick="togglePlayerMode('{player_id}')">🎥 切换为原生动图</button>
            </div>
        </div>

        <!-- 进度条滑块 -->
        <div style="background: #020408; padding: 0.4rem 1rem; border-left: 1px solid rgba(0, 229, 255, 0.2); border-right: 1px solid rgba(0, 229, 255, 0.2);">
            <input id="{player_id}-scrubber" type="range" min="0" max="{total_frames - 1}" value="0" style="width: 100%; cursor: pointer;" oninput="scrubPlayer('{player_id}', this.value)">
        </div>
        
        <div style="background: #020408; border-radius: 0 0 8px 8px; overflow: hidden; border: 1px solid rgba(0, 229, 255, 0.2); border-top: none; position: relative; margin-bottom: 1rem;">
            <!-- 逐帧高清图片播放容器 -->
            <div id="{player_id}-frame-box" style="display: block; cursor: pointer;" onclick="openLightbox(document.getElementById('{player_id}-frame-img').src, '{title}', 'REAL SEQUENCE', 'badge-real-data')">
                <img id="{player_id}-frame-img" src="{first_frame}" alt="{title}" style="width: 100%; max-height: 520px; display: block; object-fit: contain; background: #000;" onerror="this.src='{webp_src}'">
            </div>

            <!-- 原生动态 WebP 连续播放容器 (备用模式) -->
            <div id="{player_id}-webp-box" style="display: none; cursor: pointer;" onclick="openLightbox('{webp_src}', '{title}', 'REAL WEBP', 'badge-real-data')">
                <img src="{webp_src}" alt="{title}" style="width: 100%; max-height: 520px; display: block; object-fit: contain; background: #000;">
            </div>
        </div>

        <div style="background: rgba(0, 229, 255, 0.04); border-left: 3px solid #00E5FF; padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.86rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">
            {caption}
        </div>

        {phenomenon_html}
    </div>
    """
