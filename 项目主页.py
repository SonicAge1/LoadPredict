import streamlit as st
import base64

st.set_page_config(page_title="暗黑电网垄断联盟", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_video_as_page_bg(video_file):
    video_type = "video/mp4"
    video_bin = get_base64_of_bin_file(video_file)
    page_bg_video = f'''
    <video autoplay loop muted playsinline style="position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%;">
        <source src="data:{video_type};base64,{video_bin}" type="{video_type}">
    </video>
    '''
    st.markdown(page_bg_video, unsafe_allow_html=True)

# 设置背景视频
set_video_as_page_bg('background.mp4')

# CSS样式，调整文本样式和布局
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');

.content {
    font-family: 'Noto Sans SC', sans-serif;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    padding: 40px;
    background-color: rgba(0,0,0,0.5);  /* 调整为淡黑色 */
    border-radius: 10px;
    text-align: center;
    max-width: 800px;
    margin: 50px auto;
}
.title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 30px;
    color: #FFD700;
}
.subtitle {
    font-size: 24px;
    margin-bottom: 20px;
    color: #E0E0E0;
}
.text {
    font-size: 18px;
    line-height: 2;
    margin-bottom: 15px;
    color: #CCCCCC;
}
</style>
""", unsafe_allow_html=True)

# 主要内容
st.markdown("""
<div class="content">
    <h1 class="title">暗黑电网垄断联盟</h1>
    <p class="subtitle">电力负荷预测的领先企业</p>
    <p class="text">提供高效、安全和可靠的解决方案</p>
    <p class="text">技术领先的预测模型和数据分析</p>
    <p class="text">创新和优质的客户服务</p>
    <p class="text">支持客户在电力市场中取得竞争优势</p>
    <p class="text">为电力行业提供最佳的预测解决方案</p>
</div>
""", unsafe_allow_html=True)

# 添加动画效果（可选）
st.markdown("""
<script>
    document.addEventListener('DOMContentLoaded', (event) => {
        const content = document.querySelector('.content');
        content.style.opacity = 0;
        let opacity = 0;
        const fadeIn = setInterval(() => {
            if (opacity < 1) {
                opacity += 0.1;
                content.style.opacity = opacity;
            } else {
                clearInterval(fadeIn);
            }
        }, 100);
    });
</script>
""", unsafe_allow_html=True)
