"""
荧光颗粒检测器 v4.1 (Streamlit Web版)
功能：5种算法 / 量化调参 / 对象预设 / 批量处理 / 内存优化
"""

import streamlit as st
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log, blob_dog, peak_local_max
import pandas as pd
from PIL import Image, ImageDraw
import io
import gc
from datetime import datetime
import matplotlib.pyplot as plt

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="荧光颗粒检测器 v4.1",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 常量与配置 ====================
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MAX_IMAGE_SIZE = 2048  # Streamlit Cloud 1GB 内存限制下更保守

OBJECT_PRESETS = {
    '通用荧光颗粒': {
        'sensitivity': 50, 'size': 8, 'uniformity': 50, 'distance': 6,
        'algos': ['log', 'local'],
        'desc': '常规荧光标记颗粒，默认参数适用于大多数场景。'
    },
    '细胞 / 大圆颗粒': {
        'sensitivity': 60, 'size': 15, 'uniformity': 80, 'distance': 15,
        'algos': ['hough', 'log'],
        'desc': '细胞体积大、边缘较圆，霍夫圆检测对正圆细胞极稳健。'
    },
    '小点状颗粒 / 病毒样颗粒': {
        'sensitivity': 25, 'size': 3, 'uniformity': 60, 'distance': 4,
        'algos': ['dog', 'local'],
        'desc': '极小高密度颗粒，DoG速度快，适合大批量初筛。'
    },
    '致密团块 / 细胞团': {
        'sensitivity': 40, 'size': 10, 'uniformity': 20, 'distance': 10,
        'algos': ['adaptive', 'log'],
        'desc': '形状不规则、背景不均，自适应阈值+连通域效果最佳。'
    },
}

ALGO_INFO = {
    'log': {
        'name': 'LoG 斑点检测',
        'brief': '标准拉普拉斯高斯。精度最高，计算较慢，提供半径估计。',
        'has_radius': True,
    },
    'dog': {
        'name': 'DoG 快速检测',
        'brief': '高斯差分近似LoG。速度快2倍，内存占用更少，同样提供半径。',
        'has_radius': True,
    },
    'hough': {
        'name': '霍夫圆检测',
        'brief': '基于霍夫变换的圆形投票检测。对正圆颗粒/细胞极稳健。',
        'has_radius': True,
    },
    'local': {
        'name': '局部最大值',
        'brief': '仅检测亮度峰值中心。速度最快，无半径信息。',
        'has_radius': False,
    },
    'adaptive': {
        'name': '自适应阈值+连通域',
        'brief': '应对背景不均、细胞团或不规则形状。提供面积估计。',
        'has_radius': False,
    },
}

ALGO_KEYS = list(ALGO_INFO.keys())
ALGO_NAMES = [ALGO_INFO[k]['name'] for k in ALGO_KEYS]
NAME_TO_KEY = {ALGO_INFO[k]['name']: k for k in ALGO_KEYS}

# ==================== Session State 初始化 ====================
def init_session():
    defaults = {
        'preset_key': '通用荧光颗粒',
        'algo_multiselect': [ALGO_INFO[k]['name'] for k in OBJECT_PRESETS['通用荧光颗粒']['algos']],
        'sens_slider': 50,
        'size_slider': 8,
        'unif_slider': 50,
        'dist_slider': 6,
        'color_select': '红色',
        'results': [],          # 批量结果缓存
        'current_idx': 0,       # 当前查看第几张
        'has_run': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ==================== 回调函数 ====================
def apply_preset():
    """对象预设切换回调"""
    preset = OBJECT_PRESETS[st.session_state.preset_key]
    st.session_state.algo_multiselect = [ALGO_INFO[k]['name'] for k in preset['algos']]
    st.session_state.sens_slider = preset['sensitivity']
    st.session_state.size_slider = preset['size']
    st.session_state.unif_slider = preset['uniformity']
    st.session_state.dist_slider = preset['distance']

def set_sensitivity(val):
    st.session_state.sens_slider = val

# ==================== 核心函数 ====================
def load_image(file_obj):
    """加载图片，支持中文路径和TIF，带内存保护缩放"""
    file_bytes = np.frombuffer(file_obj.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("无法解码图片，可能格式不支持或文件损坏")

    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype == np.uint32:
        img = (img >> 24).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        st.toast(f"图片已缩放至 {new_w}×{new_h} 以节省内存", icon="⚠️")

    return img


def get_algorithm_params(sensitivity, size, uniformity, distance):
    """将 0-100 滑块值映射为各算法的具体参数"""
    log_thresh = 0.02 + (sensitivity / 100.0) * 0.10
    dog_thresh = 0.03 + (sensitivity / 100.0) * 0.14
    local_thresh = 0.08 + (sensitivity / 100.0) * 0.22

    adaptive_c = int(2 + (sensitivity / 100.0) * 13)
    block_size = int(5 + (20 - size) * 0.8)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    block_size = max(3, block_size)

    base_sigma = max(0.5, size * 0.4)
    if uniformity > 70:
        min_sigma = max(0.5, base_sigma * 0.8)
        max_sigma = base_sigma * 1.3
        overlap = 0.2
    elif uniformity > 30:
        min_sigma = max(0.5, base_sigma * 0.5)
        max_sigma = base_sigma * 1.8
        overlap = 0.4
    else:
        min_sigma = max(0.3, base_sigma * 0.3)
        max_sigma = base_sigma * 2.5
        overlap = 0.6

    r_px = size * 2.0
    min_area = int(3.14 * (r_px * 0.3) ** 2)
    max_area = int(3.14 * (r_px * 2.2) ** 2)
    min_area = max(4, min_area)

    hough_param2 = 8 + int(sensitivity * 0.72)
    if uniformity > 60:
        hough_min_r = max(1, int(r_px * 0.7))
        hough_max_r = int(r_px * 1.3)
    else:
        hough_min_r = max(1, int(r_px * 0.3))
        hough_max_r = int(r_px * 2.5)

    return {
        'log': {
            'min_sigma': min_sigma, 'max_sigma': max_sigma,
            'threshold': log_thresh, 'overlap': overlap, 'num_sigma': 5
        },
        'dog': {
            'min_sigma': min_sigma, 'max_sigma': max_sigma,
            'threshold': dog_thresh, 'overlap': overlap, 'num_sigma': 5
        },
        'hough': {
            'param2': hough_param2,
            'min_radius': hough_min_r,
            'max_radius': hough_max_r,
            'min_distance': distance
        },
        'local': {
            'min_distance': distance,
            'threshold_abs': local_thresh
        },
        'adaptive': {
            'block_size': block_size, 'C': adaptive_c,
            'min_area': min_area, 'max_area': max_area,
            'sigma': base_sigma
        }
    }


def run_log(channel, params):
    red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    blobs = blob_log(
        red_norm,
        min_sigma=params['min_sigma'],
        max_sigma=params['max_sigma'],
        num_sigma=params['num_sigma'],
        threshold=params['threshold'],
        overlap=params['overlap']
    )
    del red_norm
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return {'count': len(blobs), 'blobs': blobs}


def run_dog(channel, params):
    red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    blobs = blob_dog(
        red_norm,
        min_sigma=params['min_sigma'],
        max_sigma=params['max_sigma'],
        threshold=params['threshold'],
        overlap=params['overlap']
    )
    del red_norm
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return {'count': len(blobs), 'blobs': blobs}


def run_hough(channel, params):
    ch_min, ch_max = channel.min(), channel.max()
    if ch_max - ch_min < 1e-6:
        return {'count': 0, 'circles': np.array([])}

    ch_8u = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(ch_8u, (0, 0), sigmaX=1.0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=params['min_distance'],
        param1=50,
        param2=params['param2'],
        minRadius=params['min_radius'],
        maxRadius=params['max_radius']
    )

    del ch_8u, blur

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        return {'count': len(circles), 'circles': circles}
    return {'count': 0, 'circles': np.array([])}


def run_localmax(channel, params):
    bg = ndimage.uniform_filter(channel, size=50)
    red_clean = ndimage.gaussian_filter((channel - bg).clip(0), sigma=2)
    red_norm2 = (red_clean - red_clean.min()) / (red_clean.max() - red_clean.min() + 1e-8)

    coords = peak_local_max(
        red_norm2,
        min_distance=params['min_distance'],
        threshold_abs=params['threshold_abs'],
        exclude_border=True
    )
    del bg, red_clean, red_norm2
    return {'count': len(coords), 'coords': coords}


def run_adaptive(channel, params):
    ch_min, ch_max = channel.min(), channel.max()
    if ch_max - ch_min < 1e-6:
        return {'count': 0, 'centroids': np.array([]), 'areas': np.array([]), 'radii': []}

    ch_8u = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(ch_8u, (0, 0), sigmaX=params['sigma'] * 0.5)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params['block_size'],
        params['C']
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = []
    cents = []
    radii = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if params['min_area'] <= area <= params['max_area']:
            areas.append(area)
            cents.append(centroids[i])
            radii.append(np.sqrt(area / 3.1416))

    del ch_8u, blur, binary, labels, stats
    return {
        'count': len(areas),
        'centroids': np.array(cents),
        'areas': np.array(areas),
        'radii': radii
    }


def detect_all(img_bgr, color, active_keys, sens, size, uniformity, distance):
    """运行所有选中的算法"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    channel_map = {'红色': 0, '绿色': 1, '蓝色': 2}
    ch_idx = channel_map.get(color, 0)
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3:
        channel = img_rgb[:, :, ch_idx].astype(np.float32)
    else:
        channel = img_rgb.astype(np.float32)

    algo_params = get_algorithm_params(sens, size, uniformity, distance)
    results = {'img_rgb': img_rgb}

    for key in active_keys:
        if key == 'log':
            results['log'] = run_log(channel, algo_params['log'])
        elif key == 'dog':
            results['dog'] = run_dog(channel, algo_params['dog'])
        elif key == 'hough':
            results['hough'] = run_hough(channel, algo_params['hough'])
        elif key == 'local':
            results['local'] = run_localmax(channel, algo_params['local'])
        elif key == 'adaptive':
            results['adaptive'] = run_adaptive(channel, algo_params['adaptive'])

    del channel
    gc.collect()
    return results


def draw_algo_on_image(img_rgb, results, algo_key):
    """在图像上绘制指定算法的检测结果，返回 PIL Image"""
    if algo_key not in results or algo_key == 'img_rgb':
        return None

    data = results[algo_key]
    img_draw = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(img_draw)

    if algo_key in ('log', 'dog') and 'blobs' in data and len(data['blobs']) > 0:
        for y, x, r in data['blobs']:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=2)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

    elif algo_key == 'hough' and 'circles' in data and len(data['circles']) > 0:
        for x, y, r in data['circles']:
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=2)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

    elif algo_key == 'local' and 'coords' in data and len(data['coords']) > 0:
        coords = data['coords']
        for y, x in coords:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 255, 0))
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 255, 255), width=2)

    elif algo_key == 'adaptive' and 'centroids' in data and len(data['centroids']) > 0:
        cents = data['centroids']
        radii = data.get('radii', [])
        for i, (cx, cy) in enumerate(cents):
            r = int(radii[i]) if i < len(radii) else 8
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=2)
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 0, 0))

    return img_draw


def build_csv_data(results):
    """构建CSV数据"""
    rows = []
    for algo in ALGO_KEYS:
        if algo not in results:
            continue
        d = results[algo]
        if algo in ('log', 'dog'):
            for i, (y, x, r) in enumerate(d['blobs']):
                rows.append({
                    'algorithm': ALGO_INFO[algo]['name'],
                    'id': i + 1, 'x': int(x), 'y': int(y),
                    'radius': round(r, 2), 'area': round(3.1416 * r * r, 1)
                })
        elif algo == 'hough':
            for i, (x, y, r) in enumerate(d['circles']):
                rows.append({
                    'algorithm': ALGO_INFO[algo]['name'],
                    'id': i + 1, 'x': int(x), 'y': int(y),
                    'radius': int(r), 'area': round(3.1416 * r * r, 1)
                })
        elif algo == 'local':
            for i, (y, x) in enumerate(d['coords']):
                rows.append({
                    'algorithm': ALGO_INFO[algo]['name'],
                    'id': i + 1, 'x': int(x), 'y': int(y),
                    'radius': 'N/A', 'area': 'N/A'
                })
        elif algo == 'adaptive':
            for i, (cx, cy) in enumerate(d['centroids']):
                rows.append({
                    'algorithm': ALGO_INFO[algo]['name'],
                    'id': i + 1, 'x': int(cx), 'y': int(cy),
                    'radius': round(d['radii'][i], 2) if i < len(d['radii']) else 'N/A',
                    'area': int(d['areas'][i]) if i < len(d['areas']) else 'N/A'
                })
    return pd.DataFrame(rows)


# ==================== Streamlit UI ====================
st.sidebar.title("🔬 荧光颗粒检测器")
st.sidebar.caption("v4.1 Web版 | 内存优化 | 5种算法")

# ---- 对象预设 ----
st.sidebar.markdown("### 计数对象预设")
st.sidebar.selectbox(
    "选择对象类型",
    list(OBJECT_PRESETS.keys()),
    key='preset_key',
    on_change=apply_preset,
    help="切换后会自动加载推荐算法和参数"
)
st.sidebar.caption(OBJECT_PRESETS[st.session_state.preset_key]['desc'])

# ---- 算法多选 ----
st.sidebar.markdown("### 检测算法（至少选一项）")
st.sidebar.multiselect(
    "勾选要运行的算法",
    options=ALGO_NAMES,
    key='algo_multiselect',
    help="右侧结果视图将只能从已勾选的算法中选择"
)

# ---- 量化参数 ----
st.sidebar.markdown("### 量化参数调参")

# 快速跳转按钮
qcols = st.sidebar.columns(3)
qcols[0].button("高召回", on_click=set_sensitivity, args=(20,), use_container_width=True)
qcols[1].button("平衡", on_click=set_sensitivity, args=(50,), use_container_width=True)
qcols[2].button("高精度", on_click=set_sensitivity, args=(80,), use_container_width=True)

st.sidebar.slider(
    "检测严格度 (0=高召回, 100=高精度)",
    0, 100, key='sens_slider',
    help="控制所有算法的阈值严格程度"
)
st.sidebar.slider("期望颗粒尺寸级别", 1, 20, key='size_slider')
st.sidebar.slider("尺寸均匀度 (0=差异大, 100=很均匀)", 0, 100, key='unif_slider')
st.sidebar.slider("最小像素间距", 2, 30, key='dist_slider')

# ---- 颜色通道 ----
st.sidebar.markdown("### 其他参数")
st.sidebar.selectbox("颗粒颜色", ['红色', '绿色', '蓝色'], key='color_select')

# ---- 文件上传 ----
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "上传图片（支持批量）",
    type=[ext.replace('.', '') for ext in SUPPORTED_FORMATS],
    accept_multiple_files=True
)

# ---- 运行按钮 ----
st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 开始检测", type="primary", use_container_width=True)

# ==================== 主界面 ====================
st.title("荧光颗粒检测结果")

if not uploaded_files:
    st.info("👈 请在左侧上传图片并配置参数，然后点击「开始检测」")
    st.stop()

# 获取当前参数
active_names = st.session_state.algo_multiselect
active_keys = [NAME_TO_KEY[n] for n in active_names]

if not active_keys:
    st.sidebar.error("请至少选择一种检测算法！")
    st.stop()

# 结果视图选择（只能从已勾选算法中选）
st.markdown("### 结果视图选择（从已勾选算法中任选其二）")
vcols = st.columns(2)

# 确保 view_a / view_b 的选项合法
if len(active_names) >= 1:
    view_a_name = vcols[0].selectbox("左结果图", active_names, index=0, key='view_a_select')
else:
    view_a_name = None

if len(active_names) >= 2:
    view_b_name = vcols[1].selectbox("右结果图", active_names, index=min(1, len(active_names)-1), key='view_b_select')
elif len(active_names) == 1:
    view_b_name = vcols[1].selectbox("右结果图", active_names, index=0, key='view_b_select')
else:
    view_b_name = None

view_a_key = NAME_TO_KEY.get(view_a_name)
view_b_key = NAME_TO_KEY.get(view_b_name)

# ==================== 执行检测 ====================
if run_clicked:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    all_results = []
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"处理: {uploaded_file.name} ({idx+1}/{total})")
        try:
            img_bgr = load_image(uploaded_file)
            results = detect_all(
                img_bgr,
                st.session_state.color_select,
                active_keys,
                st.session_state.sens_slider,
                st.session_state.size_slider,
                st.session_state.unif_slider,
                st.session_state.dist_slider
            )
            all_results.append({
                'filename': uploaded_file.name,
                'results': results
            })
            del img_bgr
            gc.collect()
        except Exception as e:
            st.sidebar.error(f"{uploaded_file.name}: {str(e)}")
            all_results.append({
                'filename': uploaded_file.name,
                'results': None,
                'error': str(e)
            })

        progress_bar.progress((idx + 1) / total)

    st.session_state.results = all_results
    st.session_state.has_run = True
    st.session_state.current_idx = 0
    progress_bar.empty()
    status_text.empty()
    st.sidebar.success("检测完成！")

# ==================== 显示结果 ====================
if not st.session_state.has_run or not st.session_state.results:
    st.info("点击左侧「开始检测」运行分析")
    st.stop()

# 批量结果切换
all_results = st.session_state.results
if len(all_results) > 1:
    st.session_state.current_idx = st.selectbox(
        "查看结果",
        range(len(all_results)),
        format_func=lambda i: f"{i+1}. {all_results[i]['filename']}" + (" ⚠️ 错误" if all_results[i].get('error') else f" | 多算法平均: {int(np.mean([all_results[i]['results'][k]['count'] for k in active_keys if k in all_results[i]['results']]))} 个"),
        index=st.session_state.current_idx
    )

current = all_results[st.session_state.current_idx]
if current.get('error') or current['results'] is None:
    st.error(f"该图片处理失败: {current.get('error', '未知错误')}")
    st.stop()

results = current['results']
img_rgb = results['img_rgb']

# 三列展示
st.markdown(f"### 📄 {current['filename']}")
cols = st.columns(3)

# 原始图
with cols[0]:
    st.image(Image.fromarray(img_rgb), caption=f"原始图像 ({img_rgb.shape[1]}×{img_rgb.shape[0]})", width="stretch")

# 算法A
img_a = draw_algo_on_image(img_rgb, results, view_a_key) if view_a_key else None
with cols[1]:
    if img_a:
        count_a = results[view_a_key]['count'] if view_a_key in results else 0
        st.image(img_a, caption=f"{ALGO_INFO[view_a_key]['name']}: {count_a} 个颗粒", width="stretch")
    else:
        st.info("未选择算法")

# 算法B
img_b = draw_algo_on_image(img_rgb, results, view_b_key) if view_b_key else None
with cols[2]:
    if img_b:
        count_b = results[view_b_key]['count'] if view_b_key in results else 0
        st.image(img_b, caption=f"{ALGO_INFO[view_b_key]['name']}: {count_b} 个颗粒", width="stretch")
    else:
        st.info("未选择算法")

# 统计信息
st.markdown("### 统计信息")
stats_md = f"**尺寸:** {img_rgb.shape[1]} × {img_rgb.shape[0]} 像素  \n"
counts = []
for key in active_keys:
    if key in results:
        c = results[key]['count']
        stats_md += f"**{ALGO_INFO[key]['name']}:** {c} 个  \n"
        counts.append(c)

if len(counts) >= 2:
    avg = int(np.mean(counts))
    stats_md += f"**多算法平均:** <span style='color:green; font-size:1.3em'>{avg}</span> 个"

st.markdown(stats_md, unsafe_allow_html=True)

# 下载按钮
st.markdown("---")
dcols = st.columns(3)

# 下载结果图（拼接）
@st.cache_data(show_spinner=False)
def get_combined_image(img_rgb, results, view_a_key, view_b_key):
    images = [Image.fromarray(img_rgb)]
    for key in (view_a_key, view_b_key):
        img = draw_algo_on_image(img_rgb, results, key)
        if img:
            images.append(img)
    if not images:
        return None
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined

combined_img = get_combined_image(img_rgb, results, view_a_key, view_b_key)
if combined_img:
    buf = io.BytesIO()
    combined_img.save(buf, format='PNG')
    dcols[0].download_button(
        label="💾 保存结果图",
        data=buf.getvalue(),
        file_name=f"{current['filename'].split('.')[0]}_result.png",
        mime="image/png",
        use_container_width=True
    )

# 下载单张CSV
df = build_csv_data(results)
if not df.empty:
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
    dcols[1].download_button(
        label="📊 导出CSV坐标",
        data=csv_buf.getvalue().encode('utf-8-sig'),
        file_name=f"{current['filename'].split('.')[0]}_coords.csv",
        mime="text/csv",
        use_container_width=True
    )

# 批量汇总CSV
if len(all_results) > 1:
    summary = []
    for item in all_results:
        if item.get('error') or item['results'] is None:
            continue
        r = item['results']
        counts = [r[k]['count'] for k in active_keys if k in r]
        summary.append({
            'filename': item['filename'],
            **{ALGO_INFO[k]['name']: r.get(k, {}).get('count', 0) for k in ALGO_KEYS},
            'recommended': int(np.mean(counts)) if counts else 0
        })
    if summary:
        sum_df = pd.DataFrame(summary)
        sum_csv = io.StringIO()
        sum_df.to_csv(sum_csv, index=False, encoding='utf-8-sig')
        dcols[2].download_button(
            label="📥 下载批量汇总",
            data=sum_csv.getvalue().encode('utf-8-sig'),
            file_name=f"batch_summary_{datetime.now().strftime('%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

st.caption("提示：内存优化已启用，单图最大边长限制 2048px。如需处理更大图片，建议在本地运行桌面版。")
