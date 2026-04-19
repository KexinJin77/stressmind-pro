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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading

# ==========================================
# UI 增强：加载 Lottie 动画
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
# 3. 数据源模块 (🌟 极简安全的后台收集器)
# ==========================================
class SimulatedSource:
    def __init__(self): self.t = 0
    def get_next_sample(self):
        self.t += 0.1 + np.random.normal(0, 0.015) 
        return np.sin(self.t * 3.0) + np.random.normal(0, 0.05)

class CSVSource:
    def __init__(self, df):
        df.columns = [c.strip().lower() for c in df.columns]
        target_col = next((col for col in df.columns if any(k in col for k in ['ecg', 'ppg', 'mv', 'volt', 'sig', 'val', 'raw'])), None)
        self.data, self.idx = df[target_col].values, 0
    def get_next_sample(self):
        val = self.data[self.idx % len(self.data)]; self.idx += 1; return val

class HRVProcessor(VideoProcessorBase):
    def __init__(self):
        self.data_buffer = []  # 让摄像头自己在后台存数据
        self.lock = threading.Lock()
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        roi = img[h//2-40:h//2+40, w//2-40:w//2+40]
        val = np.mean(roi[:, :, 1])
        with self.lock:
            self.data_buffer.append(val)
        cv2.rectangle(img, (w//2-40, h//2-40), (w//2+50, h//2+50), (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. 呼吸引导模块
# ==========================================
def breathing_guide():
    mode = st.session_state.get("breath_mode", "4-4")
    total_cycles = 4
    steps = [("吸气", 4, "#9B5DE5"), ("呼气", 4, "#00F5D4")] if mode == "4-4" else \
            [("吸气 (鼻)", 4, "#9B5DE5"), ("屏息", 7, "#FEE440"), ("呼气 (口)", 8, "#00F5D4")]

    col1, col2 = st.columns([2, 1])
    with col2:
        lottie_breath = load_lottieurl(LOTTIE_BREATH)
        if lottie_breath: st_lottie(lottie_breath, height=300, key="breath_anim_main")
    with col1:
        ui_placeholder = st.empty()

    for cycle in range(total_cycles):
        for step_idx, (action, duration, color) in enumerate(steps):
            for time_left in range(duration, 0, -1):
                ui_placeholder.markdown(f"<div style='text-align:center; padding:60px; border-radius:30px; box-shadow: 0 8px 32px 0 rgba(155, 93, 229, 0.2); border: 3px solid {color}; background: radial-gradient(circle, {color}15, transparent);'><h1 style='color:{color}; font-size:80px;'>{action}</h1><h2 style='color:#666; font-size:45px;'>{time_left} 秒</h2><p>进度: {cycle + 1} / {total_cycles}</p></div>", unsafe_allow_html=True)
                time.sleep(1)
    st.balloons()
    st.success("✨ 训练完成！")
    time.sleep(2)
    st.session_state.stage = "setup"
    st.rerun()

# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    st.set_page_config(page_title="StressMind Pro", page_icon="🪻", layout="wide")
    st.markdown("<style>div[data-testid='stMetricValue'] { color: #9B5DE5; } .st-emotion-cache-1v0mbdj { border-radius: 15px; border: 1px solid #E6D8F8; padding: 20px; }</style>", unsafe_allow_html=True)

    if not st.session_state.get("authentication_status"):
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.markdown("<h1 style='text-align: center; color: #9B5DE5;'>🪻 StressMind Pro</h1>", unsafe_allow_html=True)
            login_tab, register_tab = st.tabs(["🔑 登录", "📝 注册"])
            with login_tab:
                authenticator.login(location="main")
                if st.session_state.get("authentication_status") is False: st.error("⚠️ 账号或密码错误")
            with register_tab:
                try:
                    email, username, name = authenticator.register_user(location='main')
                    if email:
                        with open('config.yaml', 'w', encoding='utf-8') as f: yaml.dump(config, f, allow_unicode=True)
                        st.success("注册成功，请登录")
                except Exception as e:
                    st.error(f"注册失败: {e}")
        return

    if st.session_state.get("stage") == "breathing":
        st.title("🧘‍♀️ 呼吸冥想引导")
        breathing_guide()
        st.stop()

    name = st.session_state.get("name")
    current_user = st.session_state.get("username")
    if "stage" not in st.session_state: st.session_state.stage = "setup"
    if "running" not in st.session_state: st.session_state.running = False
    if "saved_this_run" not in st.session_state: st.session_state.saved_this_run = False

    with st.sidebar:
        authenticator.logout("🚪 退出", "sidebar")
        st.subheader(f"✨ 欢迎, {name}")
        if st.button("🛑 停止/重置", use_container_width=True):
            st.session_state.running = False
            st.session_state.stage = "setup"
            st.rerun()
        if st.session_state.stage in ["setup", "measuring"]:
            mode_choice = st.selectbox("信号来源", ["实时摄像头", "模拟数据", "上传 CSV 文件"])
            duration = st.slider("时长", 15, 60, 30)
            uploaded_file = st.file_uploader("CSV", type="csv") if mode_choice == "上传 CSV 文件" else None
            if st.button("🚀 开始扫描", type="primary", use_container_width=True):
                st.session_state.active_mode, st.session_state.active_duration, st.session_state.active_file = mode_choice, duration, uploaded_file
                st.session_state.running, st.session_state.stage, st.session_state.saved_this_run = True, "measuring", False
                st.rerun()

    st.title("🪻 StressMind 负荷分析系统")

    if st.session_state.stage == "setup":
        t1, t2 = st.tabs(["🖥️ 监测中心", "📈 数据档案馆"])
        with t1:
            c1, c2 = st.columns([3, 2])
            c1.info("请在左侧配置参数，并点击『开始扫描』启动测量。")
            lottie = load_lottieurl(LOTTIE_MEDITATION)
            if lottie: 
                with c2: 
                    st_lottie(lottie, height=200)
        with t2:
            df_hist = load_history_data(current_user)
            if df_hist is not None and not df_hist.empty:
                fig = px.line(df_hist, x='timestamp', y='rmssd', markers=True)
                fig.update_traces(line_color='#9B5DE5').update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.metric("平均恢复力", f"{df_hist['rmssd'].mean():.1f} ms")
                st.subheader("📋 详细测量记录")
                st.dataframe(df_hist.sort_values(by='timestamp', ascending=False), use_container_width=True)
            else: st.warning("暂无记录")

    elif st.session_state.stage == "measuring":
        st.markdown("### 📡 实时信号采集与解析")
        plot_col, cam_col = st.columns([2, 1])
        plot_place = plot_col.empty()
        prog_bar = plot_col.progress(0)
        mode = st.session_state.active_mode

        # =========================================
        # 🌟 零卡顿的摄像头读取逻辑
        # =========================================
        if mode == "实时摄像头":
            with cam_col:
                ctx = webrtc_streamer(
                    key="hrv_cam",
                    video_processor_factory=HRVProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:global.stun.twilio.com:3478"]},{"urls": ["stun:stun.l.google.com:19302"]},{"urls": ["stun:stun.qq.com:3478"]}]}
                )
            if not ctx or not ctx.state.playing:
                plot_place.warning("👆 请先在右侧视频框中点击 'START'，并允许调用摄像头。")
                st.stop()
            else:
                if ctx.video_processor:
                    if "cam_start_time" not in st.session_state:
                        st.session_state.cam_start_time = time.time()

                    # 提取后台收集好的数据
                    with ctx.video_processor.lock:
                        current_data = list(ctx.video_processor.data_buffer)

                    total_needed = st.session_state.active_duration * 30
                    current_len = len(current_data)
                    prog_bar.progress(min(current_len / total_needed, 1.0))

                    if current_len > 10:
                        df_live = pd.DataFrame(current_data[-100:], columns=["Signal"])
                        plot_place.line_chart(df_live, color="#9B5DE5", height=300)
                    else:
                        plot_place.info("⏳ 正在稳定光电容积脉搏波信号，请保持手部平稳...")

                    if current_len >= total_needed:
                        elapsed = time.time() - st.session_state.cam_start_time
                        actual_fs = current_len / elapsed if elapsed > 0 else 30.0
                        metrics = analyze_real_hrv(current_data, actual_fs)

                        del st.session_state.cam_start_time
                        st.session_state.running = False

                        if metrics:
                            st.session_state.latest_metrics, st.session_state.stage = {**metrics, "mode": mode}, "result"
                        else:
                            st.error("数据波动过大，请保持平稳重试"); time.sleep(2); st.session_state.stage = "setup"
                        st.rerun()
                    else:
                        # 🌟 最核心的一步：只睡 0.5 秒就主动刷新页面，把控制权还给视频播放器，绝不卡顿！
                        time.sleep(0.5)
                        st.rerun()

        # =========================================
        # 模拟与CSV读取逻辑保持原样
        # =========================================
        else:
            source_obj = None
            try:
                if mode == "模拟数据": source_obj = SimulatedSource()
                elif mode == "上传 CSV 文件": source_obj = CSVSource(pd.read_csv(st.session_state.active_file))
            except Exception as e:
                st.error(f"设备启动失败: {e}"); st.session_state.stage = "setup"; st.rerun()

            data_buffer = []
            start_time = time.time()
            total_loops = st.session_state.active_duration * 30
            
            for i in range(total_loops):
                if not st.session_state.running: break
                val = source_obj.get_next_sample()
                if val is not None: data_buffer.append(val)
                
                if i % 15 == 0:
                    if len(data_buffer) > 10 and np.std(data_buffer[-10:]) > 0.001:
                        df_live = pd.DataFrame(data_buffer[-100:], columns=["Signal"])
                        plot_place.line_chart(df_live, color="#9B5DE5", height=300)
                    else:
                        plot_place.info("⏳ 正在稳定信号...")
                    prog_bar.progress((i+1)/total_loops)
                time.sleep(0.033)

            st.session_state.running = False
            if len(data_buffer) < 50:
                st.error("采集失败"); time.sleep(2); st.session_state.stage = "setup"; st.rerun()
            
            metrics = analyze_real_hrv(data_buffer, (len(data_buffer)/(time.time()-start_time)))
            if metrics:
                st.session_state.latest_metrics, st.session_state.stage = {**metrics, "mode": mode}, "result"
            else:
                st.error("数据无效"); time.sleep(2); st.session_state.stage = "setup"
            st.rerun()

    # ================== 阶段 C ==================
    elif st.session_state.stage == "result":
        res = st.session_state.latest_metrics
        rmssd, bpm = res["rmssd"], res["bpm"]
        status = "高应激" if rmssd < 25 else "状态佳"
        if not st.session_state.get("saved_this_run"):
            save_to_history(current_user, rmssd, status, res["mode"])
            st.session_state.saved_this_run = True
        st.header("📊 分析报告")
        c1, c2, c3 = st.columns(3)
        c1.metric("心率", f"{int(bpm)} BPM")
        c2.metric("RMSSD", f"{rmssd:.1f} ms", status)
        b_mode = "4-7-8" if rmssd < 25 else "4-4"
        if c3.button(f"✨ 启动 {b_mode} 呼吸", type="primary"):
            st.session_state.breath_mode, st.session_state.stage = b_mode, "breathing"; st.rerun()

if __name__ == "__main__":
    main()
