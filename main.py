import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2
import os
import yaml
import requests
from yaml.loader import SafeLoader
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, find_peaks
import streamlit_authenticator as stauth
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer
import av
import threading

# ==========================================
# UI 增强：加载 Lottie 动画 (🌟 加了缓存，彻底解决刷新消失问题)
# ==========================================
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

LOTTIE_MEDITATION = "https://assets9.lottiefiles.com/packages/lf20_puciaact.json" 
LOTTIE_BREATH = "https://assets3.lottiefiles.com/packages/lf20_5y2vpw.json"     

# ==========================================
# 1. 身份验证配置
# ==========================================
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ==========================================
# 2. 核心算法与历史存储逻辑
# ==========================================
DB_FILE = "stress_history.csv"

def save_to_history(user_id, rmssd, state, mode):
    new_data = pd.DataFrame([{"user_id": user_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rmssd": round(rmssd, 2), "state": state, "mode": mode}])
    if not os.path.isfile(DB_FILE): new_data.to_csv(DB_FILE, index=False, encoding='utf-8-sig')
    else: new_data.to_csv(DB_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def load_history_data(user_id):
    if not os.path.isfile(DB_FILE): return None
    df = pd.read_csv(DB_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last_week = datetime.now() - timedelta(days=7)
    user_df = df[(df['user_id'] == user_id) & (df['timestamp'] >= last_week)]
    return user_df.sort_values(by='timestamp')

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def analyze_real_hrv(data_buffer, actual_fs):
    if len(data_buffer) < actual_fs * 5: return None
    signal = bandpass_filter(np.array(data_buffer), fs=actual_fs)
    peaks, _ = find_peaks(signal, distance=int(actual_fs * 0.3), prominence=np.std(signal)*0.5)
    if len(peaks) < 3: return None
    valid_rr = [r for r in (np.diff(peaks) / actual_fs * 1000.0) if 300 < r < 1300]
    if len(valid_rr) < 2: return None
    bpm = 60000.0 / np.mean(valid_rr)
    rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2))
    return {"rmssd": rmssd, "bpm": bpm}

# ==========================================
# 3. 数据源模块
# ==========================================
class DataSource:
    def get_next_sample(self): raise NotImplementedError
    def release(self): pass

class SimulatedSource(DataSource):
    def __init__(self): self.t = 0
    def get_next_sample(self):
        # 🌟 修复 1：加快正弦波频率模拟正常 85 BPM 心率，并加入微小随机心跳变异
        self.t += 0.1 + np.random.normal(0, 0.015) 
        return np.sin(self.t * 3.0) + np.random.normal(0, 0.05)

class CSVSource(DataSource):
    def __init__(self, df):
        df.columns = [c.strip().lower() for c in df.columns]
        target_col = next((col for col in df.columns if any(k in col for k in ['ecg', 'ppg', 'mv', 'volt', 'sig', 'val', 'raw'])), None)
        self.data = df[target_col].values
        self.idx = 0
    def get_next_sample(self):
        val = self.data[self.idx % len(self.data)]
        self.idx += 1
        return val

# ==========================================
# 4. 用纯 Python 全局变量拯救卡死的 WebRTC 视频流
# ==========================================
global_webrtc_lock = threading.Lock()
global_green_val = None

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global global_green_val # 声明使用全局变量
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    roi = img[h//2-40:h//2+40, w//2-40:w//2+40]
    
    # 用最原始的全局变量交换数据，彻底解决崩溃问题
    with global_webrtc_lock:
        global_green_val = np.mean(roi[:, :, 1])
        
    cv2.rectangle(img, (w//2-40, h//2-40), (w//2+50, h//2+50), (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. 呼吸引导模块
# ==========================================
def breathing_guide():
    mode = st.session_state.get("breath_mode", "4-4")
    total_cycles = 4
    steps = [("吸气", 4, "#9B5DE5"), ("呼气", 4, "#00F5D4")] if mode == "4-4" else \
            [("吸气 (鼻)", 4, "#9B5DE5"), ("屏息", 7, "#FEE440"), ("呼气 (口)", 8, "#00F5D4")]

    # 动画只加载一次
    col1, col2 = st.columns([2, 1])
    with col2:
        lottie_breath = load_lottieurl(LOTTIE_BREATH)
        if lottie_breath:
            st_lottie(lottie_breath, height=300, key="breath_anim_main")
            
    with col1:
        ui_placeholder = st.empty() # 划出一块只更新不刷新的区域

    # 在内部进行倒计时，绝对不触发 st.rerun() 的全页刷新
    for cycle in range(total_cycles):
        for step_idx, (action, duration, color) in enumerate(steps):
            for time_left in range(duration, 0, -1):
                ui_placeholder.markdown(
                    f"""
                    <div style='text-align:center; padding:60px; border-radius:30px;
                                box-shadow: 0 8px 32px 0 rgba(155, 93, 229, 0.2);
                                border: 3px solid {color};
                                background: radial-gradient(circle, {color}15, transparent);
                                transition: all 0.5s ease;'>
                        <h1 style='color:{color}; font-size:80px; margin-bottom: 10px;'>{action}</h1>
                        <h2 style='color:#666; font-size:45px;'>{time_left} 秒</h2>
                        <p style='color:#999; font-size: 20px; margin-top:20px;'>进度: 第 {cycle + 1} / {total_cycles} 组</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                time.sleep(1)

    st.balloons()
    st.success("✨ 呼吸训练圆满完成！您现在的状态应该更加平静了。")
    time.sleep(2)
    st.session_state.stage = "setup"
    st.rerun()

# ==========================================
# 6. 主程序逻辑
# ==========================================
def main():
    st.set_page_config(page_title="StressMind Pro", page_icon="🪻", layout="wide")
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #9B5DE5; }
    div.stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
    .st-emotion-cache-1v0mbdj { border-radius: 15px; border: 1px solid #E6D8F8; padding: 20px; box-shadow: 0 4px 12px rgba(155, 93, 229, 0.05); }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.get("authentication_status"):
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.markdown("<h1 style='text-align: center; color: #9B5DE5;'>🪻 StressMind Pro</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888;'>您的个人数字心理疗愈舱</p>", unsafe_allow_html=True)
            login_tab, register_tab = st.tabs(["🔑 登录", "📝 新用户注册"])
            with login_tab:
                authenticator.login(location="main")
                if st.session_state.get("authentication_status") is False:
                    st.error("⚠️ 用户名或密码错误，请重试。")
            with register_tab:
                try:
                    email, username, name = authenticator.register_user(location='main')
                    if email:
                        st.balloons()
                        st.success(f"🎉 注册成功！欢迎您, {name}。")
                        st.info(f"💡 您的登录账号是：**{username}**。")
                        with open('config.yaml', 'w', encoding='utf-8') as file:
                            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
                except Exception as e:
                    st.error(f"注册失败: {e}")
        return

    name = st.session_state.get("name")
    current_user = st.session_state.get("username")
   
    # 🌟 呼吸界面的强制接管（绝对不会再漏掉）
    if st.session_state.get("stage") == "breathing":
        st.title("🧘‍♀️ 呼吸冥想引导")
        breathing_guide()
        st.stop()

    if "stage" not in st.session_state: st.session_state.stage = "setup"
    if "running" not in st.session_state: st.session_state.running = False
    if "saved_this_run" not in st.session_state: st.session_state.saved_this_run = False
    if "latest_metrics" not in st.session_state: st.session_state.latest_metrics = None

    with st.sidebar:
        authenticator.logout("🚪 退出登录", "sidebar")
        st.markdown(f"<h3 style='color:#9B5DE5;'>✨ 欢迎回来, {name}</h3>", unsafe_allow_html=True)
        st.divider()
        st.header("⚙️ 测量控制台")
        if st.button("🛑 停止 / 重置设备", use_container_width=True):
            st.session_state.running = False
            st.session_state.stage = "setup"
            st.rerun()

        st.divider()
        if st.session_state.stage in ["setup", "measuring"]:
            mode_choice = st.selectbox("📶 信号来源", ["实时摄像头", "模拟数据", "上传 CSV 文件"])
            duration = st.slider("⏱️ 采集时长 (秒)", 15, 60, 30)
            uploaded_file = None
            if mode_choice == "上传 CSV 文件":
                uploaded_file = st.file_uploader("选择信号 CSV", type="csv")

            if st.button("🚀 开始深层扫描", use_container_width=True, disabled=st.session_state.running, type="primary"):
                if mode_choice == "上传 CSV 文件" and uploaded_file is None:
                    st.error("请先上传文件")
                else:
                    st.session_state.active_mode = mode_choice
                    st.session_state.active_duration = duration
                    st.session_state.active_file = uploaded_file
                    st.session_state.running = True
                    st.session_state.stage = "measuring"
                    st.session_state.saved_this_run = False
                    st.rerun()

    st.title("🪻 StressMind 负荷分析系统")

    # ================== 阶段 A ==================
    if st.session_state.stage == "setup":
        t1, t2 = st.tabs(["🖥️ 监测中心", "📈 数据档案馆"])
        with t1:
            text_col, anim_col = st.columns([3, 2])
            with text_col:
                st.markdown(f"### 👋 您好，{name}！")
                st.write("欢迎来到您的个人健康终端。请在左侧侧边栏配置您的设备并点击**『开始深层扫描』**。")
            with anim_col:
                lottie_url = load_lottieurl(LOTTIE_MEDITATION)
                if lottie_url: st_lottie(lottie_url, height=200, key="meditation_anim")

        with t2:
            df_hist = load_history_data(current_user)
            if df_hist is not None and not df_hist.empty:
                st.subheader(f"📊 {name} 的周度恢复力追踪")
                fig_hist = px.line(df_hist, x='timestamp', y='rmssd', markers=True)
                fig_hist.update_traces(line_color='#9B5DE5', line_width=3, marker_size=8)
                fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                c1, c2 = st.columns(2)
                c1.metric("周平均恢复力", f"{df_hist['rmssd'].mean():.1f} ms", "较上周评估")
                c2.metric("高压预警次数", f"{len(df_hist[df_hist['state'] == '高应激'])} 次", delta_color="inverse")
                
                # 🌟 修复：把缺失的表格原原本本补回这里！
                st.markdown("#### 📋 详细测量记录")
                st.dataframe(df_hist.sort_values(by='timestamp', ascending=False), use_container_width=True)
            else:
                st.warning("📭 暂无记录。完成您的第一次生命体征采集后，数据报告将在此生成。")

    # ================== 阶段 B ==================
    elif st.session_state.stage == "measuring":
        st.markdown("### 📡 实时信号采集与解析")
        plot_col, cam_col = st.columns([2, 1])
        data_buffer, plot_place, prog_bar = [], plot_col.empty(), plot_col.progress(0)
        mode = st.session_state.active_mode
        source_obj, ctx = None, None

        if mode == "实时摄像头":
            with cam_col:
                ctx = webrtc_streamer(
                    key="hrv_cam",
                    video_frame_callback=video_frame_callback,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
            if not ctx or not ctx.state.playing:
                plot_place.warning("👆 请先在右侧视频框中点击 'START' 允许调用摄像头。")
                st.stop()
        else:
            try:
                if mode == "模拟数据": source_obj = SimulatedSource()
                elif mode == "上传 CSV 文件": source_obj = CSVSource(pd.read_csv(st.session_state.active_file))
            except Exception as e:
                st.error(f"设备启动失败: {e}"); st.session_state.stage = "setup"; st.rerun()

        start_time = time.time()
        total_loops = st.session_state.active_duration * 30
        
        for i in range(total_loops):
            if not st.session_state.running or st.session_state.stage != "measuring": break
            val = None
            if mode == "实时摄像头":
                if ctx and ctx.state.playing:
                    # 获取纯净全局变量数据，不卡顿
                    with global_webrtc_lock:
                        val = global_green_val
            else:
                val = source_obj.get_next_sample()
                
            if val is not None: data_buffer.append(val)
            
            if i % 10 == 0:
                if len(data_buffer) > 10 and np.std(data_buffer[-10:]) > 0.001:
                    y_data = data_buffer[-100:]
                    x_data = list(range(len(y_data)))
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line=dict(color='#9B5DE5', width=2.5, shape='spline'), fill='tozeroy', fillcolor='rgba(155, 93, 229, 0.15)'))
                    fig_live.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
                    plot_place.plotly_chart(fig_live, use_container_width=True)
                else:
                    plot_place.info("⏳ 正在校准神经元信号...")
                prog_bar.progress((i+1)/total_loops)
                
            time.sleep(0.033)
        
        if source_obj: source_obj.release()
        st.session_state.running = False
       # 🌟 3. HRV 结算
        if st.session_state.stage == "measuring":
            if len(data_buffer) < 50:
                st.error("⚠️ 未能采集到有效的生命体征信号。请确保视频画面出现后，再耐心等待几秒。")
                time.sleep(2.5) # 🌟 修复 2：停留两秒让用户看清报错
                st.session_state.stage = "setup"
                st.rerun()
            else:
                elapsed = time.time() - start_time
                actual_fs = len(data_buffer) / elapsed if (elapsed > 0 and len(data_buffer) > 0) else 30.0
                metrics = analyze_real_hrv(data_buffer, actual_fs)
                
                if metrics:
                    metrics["mode"] = mode
                    st.session_state.latest_metrics = metrics
                    st.session_state.stage = "result"
                else:
                    st.error("⚠️ 信号噪音过大或发生剧烈抖动。请保持双手平稳重试。")
                    time.sleep(2.5) # 🌟 修复 2：停留两秒让用户看清报错
                    st.session_state.stage = "setup"
                st.rerun()

    # ================== 阶段 C ==================
    elif st.session_state.stage == "result":
        res = st.session_state.latest_metrics
        rmssd, bpm = res["rmssd"], res["bpm"]
        status = "高应激" if rmssd < 25 else "状态佳"
        
        if not st.session_state.saved_this_run:
            save_to_history(current_user, rmssd, status, res["mode"])
            st.session_state.saved_this_run = True

        st.markdown("### 📋 深层生命体征分析报告")
        st.divider()
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1: st.metric("静息心率 (BPM)", f"{int(bpm)} 次/分")
        with c2: st.metric("心率变异性", f"{rmssd:.1f} ms", status)
        with c3:
            b_mode = "4-7-8" if rmssd < 25 else "4-4"
            st.info(f"🧘‍♀️ **自律神经恢复建议**\n\n系统为您匹配了 **{b_mode}** 训练。")
            if st.button(f"✨ 立即启动呼吸", type="primary", use_container_width=True):
                st.session_state.breath_mode = b_mode
                st.session_state.stage = "breathing"; st.rerun()

if __name__ == "__main__":
    main()
