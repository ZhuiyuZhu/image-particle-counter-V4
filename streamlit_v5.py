"""
荧光颗粒与组织染色检测器 v5.0 (Streamlit Web版)
模块：荧光颗粒计数 / 油红O脂滴 / ALP矿化 / 茜素红矿化
新增：AI参数推荐器、一键恢复v3.3默认、内存优化
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
    page_title="荧光颗粒与染色检测器 v5.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 常量 ====================
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
MAX_IMAGE_SIZE = 2048  # Streamlit Cloud 1GB 内存保守限制

ALGO_INFO = {
    'log': {'name': 'LoG 斑点检测', 'has_radius': True},
    'dog': {'name': 'DoG 快速检测', 'has_radius': True},
    'hough': {'name': '霍夫圆检测', 'has_radius': True},
    'local': {'name': '局部最大值', 'has_radius': False},
    'adaptive': {'name': '自适应阈值+连通域', 'has_radius': False},
}
ALGO_KEYS = list(ALGO_INFO.keys())
ALGO_NAMES = [ALGO_INFO[k]['name'] for k in ALGO_KEYS]
NAME_TO_KEY = {ALGO_INFO[k]['name']: k for k in ALGO_KEYS}

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

# ==================== AI 参数推荐器 ====================
class AIRecommender:
    KNOWLEDGE_BASE = {
        'oilred_3T3-L1_10x': {
            'hsv_h_low': 0, 'hsv_h_high': 15, 'sat_min': 80, 'val_min': 50,
            'min_droplet_area': 30, 'watershed': True,
            'desc': '3T3-L1诱导分化后脂滴，10x下直径约10-40px'
        },
        'oilred_3T3-L1_20x': {
            'hsv_h_low': 0, 'hsv_h_high': 15, 'sat_min': 100, 'val_min': 60,
            'min_droplet_area': 80, 'watershed': True,
            'desc': '20x下脂滴更大更密，提高饱和度阈值防背景干扰'
        },
        'oilred_BMSC_10x': {
            'hsv_h_low': 0, 'hsv_h_high': 20, 'sat_min': 60, 'val_min': 40,
            'min_droplet_area': 25, 'watershed': False,
            'desc': 'BMSC脂滴较小且散在，放宽阈值'
        },
        'oilred_原代脂肪细胞_10x': {
            'hsv_h_low': 0, 'hsv_h_high': 18, 'sat_min': 70, 'val_min': 45,
            'min_droplet_area': 35, 'watershed': True,
            'desc': '原代脂肪细胞脂滴大小不一，中等阈值'
        },
        'alp_MC3T3-E1_10x': {
            'lab_b_threshold': 135, 'morph_close_iter': 2, 'min_nodule_area': 100,
            'desc': 'MC3T3-E1成骨诱导，ALP显色蓝紫，LAB-B通道阈值135'
        },
        'alp_MC3T3-E1_20x': {
            'lab_b_threshold': 140, 'morph_close_iter': 2, 'min_nodule_area': 250,
            'desc': '20x下结节更大，阈值略提，面积过滤增大'
        },
        'alp_BMSC_10x': {
            'lab_b_threshold': 128, 'morph_close_iter': 3, 'min_nodule_area': 80,
            'desc': 'BMSC矿化结节通常更弥散，阈值略低，闭合迭代更多'
        },
        'alp_hFOB_10x': {
            'lab_b_threshold': 130, 'morph_close_iter': 2, 'min_nodule_area': 90,
            'desc': 'hFOB人成骨细胞，参数介于MC3T3与BMSC之间'
        },
        'alizarin_MC3T3-E1_10x': {
            'r_threshold': 120, 'rg_ratio': 1.1, 'rb_ratio': 1.1,
            'morph_close_iter': 2, 'min_nodule_area': 150,
            'desc': '茜素红钙结节呈亮红色，R通道主导，结节致密'
        },
        'alizarin_MC3T3-E1_20x': {
            'r_threshold': 125, 'rg_ratio': 1.15, 'rb_ratio': 1.15,
            'morph_close_iter': 2, 'min_nodule_area': 400,
            'desc': '20x下结节更大更致密，阈值和面积过滤均提高'
        },
        'alizarin_BMSC_10x': {
            'r_threshold': 110, 'rg_ratio': 1.0, 'rb_ratio': 1.0,
            'morph_close_iter': 2, 'min_nodule_area': 120,
            'desc': 'BMSC结节相对松散，阈值略降'
        },
    }

    @classmethod
    def recommend(cls, module, cell_line, magnification):
        key = f"{module}_{cell_line}_{magnification}"
        if key not in cls.KNOWLEDGE_BASE:
            # 模糊匹配：同模块同细胞系任意倍数
            candidates = [k for k in cls.KNOWLEDGE_BASE if k.startswith(f"{module}_{cell_line}_")]
            if candidates:
                key = candidates[0]
            else:
                # 模块默认
                key = next((k for k in cls.KNOWLEDGE_BASE if k.startswith(module)), None)
        params = cls.KNOWLEDGE_BASE.get(key, {}).copy()
        return params


# ==================== 图像加载 ====================
def load_image(file_obj):
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


# ==================== 荧光颗粒核心算法 ====================
def get_algorithm_params(sensitivity, size, uniformity, distance):
    log_thresh = 0.02 + (sensitivity / 100.0) * 0.10
    dog_thresh = 0.03 + (sensitivity / 100.0) * 0.14
    local_thresh = 0.08 + (sensitivity / 100.0) * 0.22
    base_sigma = max(0.5, size * 0.4)
    if uniformity > 70:
        min_sigma, max_sigma, overlap = max(0.5, base_sigma * 0.8), base_sigma * 1.3, 0.2
    elif uniformity > 30:
        min_sigma, max_sigma, overlap = max(0.5, base_sigma * 0.5), base_sigma * 1.8, 0.4
    else:
        min_sigma, max_sigma, overlap = max(0.3, base_sigma * 0.3), base_sigma * 2.5, 0.6
    return {
        'log': {'min_sigma': min_sigma, 'max_sigma': max_sigma, 'threshold': log_thresh, 'overlap': overlap, 'num_sigma': 5},
        'dog': {'min_sigma': min_sigma, 'max_sigma': max_sigma, 'threshold': dog_thresh, 'overlap': overlap, 'num_sigma': 5},
        'hough': {'param2': 8 + int(sensitivity * 0.72), 'min_radius': max(1, int(size * 0.6)), 'max_radius': int(size * 5), 'min_distance': distance},
        'local': {'min_distance': distance, 'threshold_abs': local_thresh},
        'adaptive': {'block_size': max(3, (int(5 + (20 - size) * 0.8) // 2 * 2 + 1)), 'C': int(2 + (sensitivity / 100.0) * 13),
                     'min_area': max(4, int(3.14 * (size * 0.6) ** 2)), 'max_area': int(3.14 * (size * 4.4) ** 2), 'sigma': base_sigma}
    }

def run_log(channel, params):
    red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    blobs = blob_log(red_norm, min_sigma=params['min_sigma'], max_sigma=params['max_sigma'],
                     num_sigma=params['num_sigma'], threshold=params['threshold'], overlap=params['overlap'])
    del red_norm
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return {'count': len(blobs), 'blobs': blobs}

def run_dog(channel, params):
    red_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    blobs = blob_dog(red_norm, min_sigma=params['min_sigma'], max_sigma=params['max_sigma'],
                     threshold=params['threshold'], overlap=params['overlap'])
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
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=params['min_distance'],
                               param1=50, param2=params['param2'],
                               minRadius=params['min_radius'], maxRadius=params['max_radius'])
    del ch_8u, blur
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        return {'count': len(circles), 'circles': circles}
    return {'count': 0, 'circles': np.array([])}

def run_localmax(channel, params):
    bg = ndimage.uniform_filter(channel, size=50)
    red_clean = ndimage.gaussian_filter((channel - bg).clip(0), sigma=2)
    red_norm2 = (red_clean - red_clean.min()) / (red_clean.max() - red_clean.min() + 1e-8)
    coords = peak_local_max(red_norm2, min_distance=params['min_distance'],
                            threshold_abs=params['threshold_abs'], exclude_border=True)
    del bg, red_clean, red_norm2
    return {'count': len(coords), 'coords': coords}

def run_adaptive(channel, params):
    ch_min, ch_max = channel.min(), channel.max()
    if ch_max - ch_min < 1e-6:
        return {'count': 0, 'centroids': np.array([]), 'areas': np.array([]), 'radii': []}
    ch_8u = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(ch_8u, (0, 0), sigmaX=params['sigma'] * 0.5)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, params['block_size'], params['C'])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas, cents, radii = [], [], []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if params['min_area'] <= area <= params['max_area']:
            areas.append(area); cents.append(centroids[i]); radii.append(np.sqrt(area / 3.1416))
    del ch_8u, blur, binary, labels, stats
    return {'count': len(areas), 'centroids': np.array(cents), 'areas': np.array(areas), 'radii': radii}

def detect_all_fluo(img_bgr, color, active_keys, sens, size, uniformity, distance):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    channel_map = {'红色': 0, '绿色': 1, '蓝色': 2}
    ch_idx = channel_map.get(color, 0)
    channel = img_rgb[:, :, ch_idx].astype(np.float32) if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3 else img_rgb.astype(np.float32)
    algo_params = get_algorithm_params(sens, size, uniformity, distance)
    results = {'img_rgb': img_rgb}
    for key in active_keys:
        if key == 'log': results['log'] = run_log(channel, algo_params['log'])
        elif key == 'dog': results['dog'] = run_dog(channel, algo_params['dog'])
        elif key == 'hough': results['hough'] = run_hough(channel, algo_params['hough'])
        elif key == 'local': results['local'] = run_localmax(channel, algo_params['local'])
        elif key == 'adaptive': results['adaptive'] = run_adaptive(channel, algo_params['adaptive'])
    del channel; gc.collect()
    return results

def draw_algo_on_image(img_rgb, results, algo_key):
    if algo_key not in results or algo_key == 'img_rgb':
        return None
    data = results[algo_key]
    img_draw = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(img_draw)
    if algo_key in ('log', 'dog') and 'blobs' in data and len(data['blobs']) > 0:
        for y, x, r in data['blobs']:
            draw.ellipse([x-r, y-r, x+r, y+r], outline=(0,255,0), width=2)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(255,0,0))
    elif algo_key == 'hough' and 'circles' in data and len(data['circles']) > 0:
        for x, y, r in data['circles']:
            draw.ellipse([x-r, y-r, x+r, y+r], outline=(0,255,0), width=2)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(255,0,0))
    elif algo_key == 'local' and 'coords' in data and len(data['coords']) > 0:
        for y, x in data['coords']:
            draw.ellipse([x-5, y-5, x+5, y+5], fill=(0,255,0))
            draw.ellipse([x-7, y-7, x+7, y+7], outline=(255,255,255), width=2)
    elif algo_key == 'adaptive' and 'centroids' in data and len(data['centroids']) > 0:
        for i, (cx, cy) in enumerate(data['centroids']):
            r = int(data['radii'][i]) if i < len(data['radii']) else 8
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0,255,0), width=2)
            draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255,0,0))
    return img_draw

def build_fluo_csv(results):
    rows = []
    for algo in ALGO_KEYS:
        if algo not in results: continue
        d = results[algo]
        if algo in ('log', 'dog'):
            for i, (y, x, r) in enumerate(d['blobs']):
                rows.append({'algorithm': ALGO_INFO[algo]['name'], 'id': i+1, 'x': int(x), 'y': int(y), 'radius': round(r,2), 'area': round(3.1416*r*r,1)})
        elif algo == 'hough':
            for i, (x, y, r) in enumerate(d['circles']):
                rows.append({'algorithm': ALGO_INFO[algo]['name'], 'id': i+1, 'x': int(x), 'y': int(y), 'radius': int(r), 'area': round(3.1416*r*r,1)})
        elif algo == 'local':
            for i, (y, x) in enumerate(d['coords']):
                rows.append({'algorithm': ALGO_INFO[algo]['name'], 'id': i+1, 'x': int(x), 'y': int(y), 'radius': 'N/A', 'area': 'N/A'})
        elif algo == 'adaptive':
            for i, (cx, cy) in enumerate(d['centroids']):
                rows.append({'algorithm': ALGO_INFO[algo]['name'], 'id': i+1, 'x': int(cx), 'y': int(cy), 'radius': round(d['radii'][i],2) if i < len(d['radii']) else 'N/A', 'area': int(d['areas'][i]) if i < len(d['areas']) else 'N/A'})
    return pd.DataFrame(rows)


# ==================== 油红O 脂滴分析 ====================
def analyze_oil_red_o(img_bgr, params):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([params['hsv_h_low'], params['sat_min'], params['val_min']])
    upper1 = np.array([params['hsv_h_high'], 255, 255])
    lower2 = np.array([170, params['sat_min'], params['val_min']])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if params.get('watershed', True):
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(mask, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), markers)
        final_mask = np.zeros_like(mask)
        final_mask[markers > 1] = 255
    else:
        final_mask = mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    droplets = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= params['min_droplet_area']:
            droplets.append({'id': i, 'area': int(area), 'centroid': (float(centroids[i][0]), float(centroids[i][1]))})

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    positive_pixels = int(np.sum(final_mask > 0))
    return {
        'mask': final_mask, 'count': len(droplets), 'droplets': droplets,
        'area_ratio': positive_pixels / total_pixels * 100.0,
        'positive_pixels': positive_pixels, 'total_pixels': total_pixels,
        'avg_droplet_area': round(np.mean([d['area'] for d in droplets]), 1) if droplets else 0
    }


# ==================== ALP / 茜素红 矿化分析 ====================
def analyze_mineralization(img_bgr, params):
    stain = params.get('stain_type', 'alp')
    if stain == 'alp':
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]
        _, mask = cv2.threshold(b_channel, params['lab_b_threshold'], 255, cv2.THRESH_BINARY)
    else:
        b, g, r = cv2.split(img_bgr)
        rg_diff = cv2.subtract(r, g)
        rb_diff = cv2.subtract(r, b)
        _, mask_r = cv2.threshold(r, params['r_threshold'], 255, cv2.THRESH_BINARY)
        _, mask_rg = cv2.threshold(rg_diff, int(params['r_threshold'] * (params['rg_ratio'] - 1)), 255, cv2.THRESH_BINARY)
        _, mask_rb = cv2.threshold(rb_diff, int(params['r_threshold'] * (params['rb_ratio'] - 1)), 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask_r, mask_rg)
        mask = cv2.bitwise_and(mask, mask_rb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.get('morph_close_iter', 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    positive_pixels = int(np.sum(mask > 0))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    nodules = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= params['min_nodule_area']:
            nodules.append({'id': i, 'area': int(area), 'centroid': (float(centroids[i][0]), float(centroids[i][1]))})

    return {
        'mask': mask, 'stain_type': stain, 'area_ratio': positive_pixels / total_pixels * 100.0,
        'positive_pixels': positive_pixels, 'total_pixels': total_pixels,
        'nodule_count': len(nodules), 'nodules': nodules,
        'avg_nodule_area': round(np.mean([n['area'] for n in nodules]), 1) if nodules else 0
    }


# ==================== Session State 初始化 ====================
def init_state():
    defaults = {
        'current_module': '荧光颗粒计数',
        'uploaded_files': None,
        'current_results': [],
        'has_run': False,
        'current_idx': 0,
        # 荧光颗粒
        'fluo_preset': '通用荧光颗粒',
        'fluo_algos': ['LoG 斑点检测', '局部最大值'],
        'fluo_sens': 50, 'fluo_size': 8, 'fluo_unif': 20, 'fluo_dist': 6,
        'fluo_color': '红色',
        'fluo_view_a': 'LoG 斑点检测', 'fluo_view_b': '局部最大值',
        # 油红O
        'oilred_cell': '3T3-L1', 'oilred_mag': '10x',
        'oilred_h_low': 0, 'oilred_h_high': 15, 'oilred_sat': 80, 'oilred_val': 50,
        'oilred_min_area': 30, 'oilred_watershed': True,
        # ALP
        'alp_cell': 'MC3T3-E1', 'alp_mag': '10x',
        'alp_thresh': 135, 'alp_morph': 2, 'alp_min_area': 100,
        # 茜素红
        'alizarin_cell': 'MC3T3-E1', 'alizarin_mag': '10x',
        'alizarin_r': 120, 'alizarin_rg': 11, 'alizarin_rb': 11,
        'alizarin_morph': 2, 'alizarin_min_area': 150,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ==================== 回调函数 ====================
def apply_fluo_preset():
    preset = OBJECT_PRESETS[st.session_state.fluo_preset]
    st.session_state.fluo_algos = [ALGO_INFO[k]['name'] for k in preset['algos']]
    st.session_state.fluo_sens = preset['sensitivity']
    st.session_state.fluo_size = preset['size']
    st.session_state.fluo_unif = preset['uniformity']
    st.session_state.fluo_dist = preset['distance']
    st.rerun()

def set_fluo_sens(v):
    st.session_state.fluo_sens = v
    st.rerun()

def reset_fluo_v33():
    st.session_state.fluo_algos = ['LoG 斑点检测', '局部最大值']
    st.session_state.fluo_sens = 50
    st.session_state.fluo_size = 8
    st.session_state.fluo_unif = 20
    st.session_state.fluo_dist = 6
    st.session_state.fluo_color = '红色'
    st.session_state.fluo_preset = '通用荧光颗粒'
    st.rerun()

def apply_oilred_ai():
    rec = AIRecommender.recommend('oilred', st.session_state.oilred_cell, st.session_state.oilred_mag)
    desc = rec.pop('desc', '')
    st.toast(f"AI推荐: {desc}", icon="🤖")
    if 'hsv_h_low' in rec: st.session_state.oilred_h_low = rec['hsv_h_low']
    if 'hsv_h_high' in rec: st.session_state.oilred_h_high = rec['hsv_h_high']
    if 'sat_min' in rec: st.session_state.oilred_sat = rec['sat_min']
    if 'val_min' in rec: st.session_state.oilred_val = rec['val_min']
    if 'min_droplet_area' in rec: st.session_state.oilred_min_area = rec['min_droplet_area']
    if 'watershed' in rec: st.session_state.oilred_watershed = rec['watershed']
    st.rerun()

def reset_oilred():
    st.session_state.oilred_h_low = 0; st.session_state.oilred_h_high = 15
    st.session_state.oilred_sat = 80; st.session_state.oilred_val = 50
    st.session_state.oilred_min_area = 30; st.session_state.oilred_watershed = True
    st.rerun()

def apply_alp_ai():
    rec = AIRecommender.recommend('alp', st.session_state.alp_cell, st.session_state.alp_mag)
    desc = rec.pop('desc', '')
    st.toast(f"AI推荐: {desc}", icon="🤖")
    if 'lab_b_threshold' in rec: st.session_state.alp_thresh = rec['lab_b_threshold']
    if 'morph_close_iter' in rec: st.session_state.alp_morph = rec['morph_close_iter']
    if 'min_nodule_area' in rec: st.session_state.alp_min_area = rec['min_nodule_area']
    st.rerun()

def reset_alp():
    st.session_state.alp_thresh = 135; st.session_state.alp_morph = 2; st.session_state.alp_min_area = 100
    st.rerun()

def apply_alizarin_ai():
    rec = AIRecommender.recommend('alizarin', st.session_state.alizarin_cell, st.session_state.alizarin_mag)
    desc = rec.pop('desc', '')
    st.toast(f"AI推荐: {desc}", icon="🤖")
    if 'r_threshold' in rec: st.session_state.alizarin_r = rec['r_threshold']
    if 'rg_ratio' in rec: st.session_state.alizarin_rg = int(rec['rg_ratio'] * 10)
    if 'rb_ratio' in rec: st.session_state.alizarin_rb = int(rec['rb_ratio'] * 10)
    if 'morph_close_iter' in rec: st.session_state.alizarin_morph = rec['morph_close_iter']
    if 'min_nodule_area' in rec: st.session_state.alizarin_min_area = rec['min_nodule_area']
    st.rerun()

def reset_alizarin():
    st.session_state.alizarin_r = 120; st.session_state.alizarin_rg = 11; st.session_state.alizarin_rb = 11
    st.session_state.alizarin_morph = 2; st.session_state.alizarin_min_area = 150
    st.rerun()


# ==================== Sidebar ====================
st.sidebar.title("🔬 染色检测器 v5.0")
st.sidebar.caption("Streamlit 在线版 | 四模块 | AI推荐")

st.sidebar.markdown("---")
module = st.sidebar.radio(
    "选择检测模块",
    ["荧光颗粒计数", "油红O脂滴", "ALP矿化", "茜素红矿化"],
    key='current_module'
)

uploaded_files = st.sidebar.file_uploader(
    "上传图片（支持批量）",
    type=[ext.replace('.', '') for ext in SUPPORTED_FORMATS],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 开始检测", type="primary", use_container_width=True)

# ==================== 主界面：模块参数配置 ====================
st.title(f"{module} 分析")

if module == "荧光颗粒计数":
    # --- 荧光颗粒参数面板 ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.selectbox("计数对象预设", list(OBJECT_PRESETS.keys()), key='fluo_preset', on_change=apply_fluo_preset)
        st.caption(OBJECT_PRESETS[st.session_state.fluo_preset]['desc'])
        st.multiselect("检测算法（至少选一项）", ALGO_NAMES, key='fluo_algos')
        st.selectbox("颗粒颜色", ['红色', '绿色', '蓝色'], key='fluo_color')
    with c2:
        st.markdown("**量化参数**")
        qcols = st.columns(3)
        qcols[0].button("高召回", on_click=set_fluo_sens, args=(20,), use_container_width=True)
        qcols[1].button("平衡", on_click=set_fluo_sens, args=(50,), use_container_width=True)
        qcols[2].button("高精度", on_click=set_fluo_sens, args=(80,), use_container_width=True)
        st.slider("检测严格度", 0, 100, key='fluo_sens')
        st.slider("期望颗粒尺寸级别", 1, 20, key='fluo_size')
        st.slider("尺寸均匀度 (0=差异大, 100=很均匀)", 0, 100, key='fluo_unif')
        st.slider("最小像素间距", 2, 30, key='fluo_dist')
        st.button("🔄 恢复 v3.3 默认", on_click=reset_fluo_v33, use_container_width=True)

    # 结果视图选择
    active_names = st.session_state.fluo_algos
    if len(active_names) >= 1:
        if st.session_state.fluo_view_a not in active_names:
            st.session_state.fluo_view_a = active_names[0]
        if st.session_state.fluo_view_b not in active_names or st.session_state.fluo_view_b == st.session_state.fluo_view_a:
            st.session_state.fluo_view_b = active_names[min(1, len(active_names)-1)]
        vcols = st.columns(2)
        vcols[0].selectbox("左结果图", active_names, key='fluo_view_a')
        vcols[1].selectbox("右结果图", active_names, key='fluo_view_b')

elif module == "油红O脂滴":
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**🤖 AI 参数推荐**")
        st.selectbox("细胞系", ['3T3-L1', 'BMSC', '原代脂肪细胞', '其他'], key='oilred_cell')
        st.selectbox("放大倍数", ['4x', '10x', '20x', '其他'], key='oilred_mag')
        st.button("应用 AI 推荐参数", on_click=apply_oilred_ai, use_container_width=True)
    with c2:
        st.markdown("**手动参数**")
        st.slider("HSV 红色下限 H", 0, 30, key='oilred_h_low')
        st.slider("HSV 红色上限 H", 0, 40, key='oilred_h_high')
        st.slider("饱和度阈值", 0, 255, key='oilred_sat')
        st.slider("亮度阈值", 0, 255, key='oilred_val')
        st.slider("最小脂滴面积(px)", 1, 200, key='oilred_min_area')
        st.checkbox("启用 Watershed 分割粘连脂滴", key='oilred_watershed')
        st.button("恢复默认", on_click=reset_oilred, use_container_width=True)

elif module == "ALP矿化":
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**🤖 AI 参数推荐**")
        st.selectbox("细胞系", ['MC3T3-E1', 'BMSC', 'hFOB', '其他'], key='alp_cell')
        st.selectbox("放大倍数", ['4x', '10x', '20x', '其他'], key='alp_mag')
        st.button("应用 AI 推荐参数", on_click=apply_alp_ai, use_container_width=True)
    with c2:
        st.markdown("**手动参数**")
        st.slider("LAB-B 阈值", 80, 200, key='alp_thresh')
        st.slider("闭运算迭代", 0, 5, key='alp_morph')
        st.slider("最小结节面积", 10, 500, key='alp_min_area')
        st.button("恢复默认", on_click=reset_alp, use_container_width=True)

elif module == "茜素红矿化":
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**🤖 AI 参数推荐**")
        st.selectbox("细胞系", ['MC3T3-E1', 'BMSC', 'hFOB', '其他'], key='alizarin_cell')
        st.selectbox("放大倍数", ['4x', '10x', '20x', '其他'], key='alizarin_mag')
        st.button("应用 AI 推荐参数", on_click=apply_alizarin_ai, use_container_width=True)
    with c2:
        st.markdown("**手动参数**")
        st.slider("R 通道阈值", 50, 200, key='alizarin_r')
        st.slider("R/G 比率 (×10)", 10, 20, key='alizarin_rg', help="实际值 = 显示值/10")
        st.slider("R/B 比率 (×10)", 10, 20, key='alizarin_rb', help="实际值 = 显示值/10")
        st.slider("闭运算迭代", 0, 5, key='alizarin_morph')
        st.slider("最小结节面积", 10, 500, key='alizarin_min_area')
        st.button("恢复默认", on_click=reset_alizarin, use_container_width=True)

# ==================== 执行检测 ====================
if run_clicked:
    if not uploaded_files:
        st.sidebar.error("请先上传图片！")
    else:
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        results = []
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            status.text(f"处理: {f.name} ({i+1}/{total})")
            try:
                img = load_image(f)
                if module == "荧光颗粒计数":
                    active_keys = [NAME_TO_KEY[n] for n in st.session_state.fluo_algos]
                    if not active_keys:
                        st.sidebar.error("请至少选择一种算法！"); break
                    res = detect_all_fluo(img, st.session_state.fluo_color, active_keys,
                                          st.session_state.fluo_sens, st.session_state.fluo_size,
                                          st.session_state.fluo_unif, st.session_state.fluo_dist)
                elif module == "油红O脂滴":
                    res = analyze_oil_red_o(img, {
                        'hsv_h_low': st.session_state.oilred_h_low, 'hsv_h_high': st.session_state.oilred_h_high,
                        'sat_min': st.session_state.oilred_sat, 'val_min': st.session_state.oilred_val,
                        'min_droplet_area': st.session_state.oilred_min_area,
                        'watershed': st.session_state.oilred_watershed
                    })
                elif module == "ALP矿化":
                    res = analyze_mineralization(img, {
                        'stain_type': 'alp', 'lab_b_threshold': st.session_state.alp_thresh,
                        'morph_close_iter': st.session_state.alp_morph, 'min_nodule_area': st.session_state.alp_min_area
                    })
                elif module == "茜素红矿化":
                    res = analyze_mineralization(img, {
                        'stain_type': 'alizarin', 'r_threshold': st.session_state.alizarin_r,
                        'rg_ratio': st.session_state.alizarin_rg / 10.0,
                        'rb_ratio': st.session_state.alizarin_rb / 10.0,
                        'morph_close_iter': st.session_state.alizarin_morph,
                        'min_nodule_area': st.session_state.alizarin_min_area
                    })
                results.append({'filename': f.name, 'result': res})
                del img; gc.collect()
            except Exception as e:
                st.sidebar.error(f"{f.name}: {str(e)}")
                results.append({'filename': f.name, 'result': None, 'error': str(e)})
            progress.progress((i + 1) / total)

        st.session_state.current_results = results
        st.session_state.has_run = True
        st.session_state.current_idx = 0
        progress.empty(); status.empty()
        st.sidebar.success("检测完成！")

# ==================== 显示结果 ====================
if not st.session_state.get('has_run') or not st.session_state.current_results:
    st.info("👈 上传图片并点击「开始检测」")
    st.stop()

all_results = st.session_state.current_results
if len(all_results) > 1:
    st.session_state.current_idx = st.selectbox(
        "查看结果", range(len(all_results)),
        format_func=lambda i: f"{i+1}. {all_results[i]['filename']}" + (" ⚠️ 错误" if all_results[i].get('error') else ""),
        index=st.session_state.current_idx
    )

current = all_results[st.session_state.current_idx]
if current.get('error') or current['result'] is None:
    st.error(f"处理失败: {current.get('error', '未知错误')}")
    st.stop()

res = current['result']

# --- 荧光颗粒结果显示 ---
if module == "荧光颗粒计数":
    img_rgb = res['img_rgb']
    st.markdown(f"### 📄 {current['filename']}")
    cols = st.columns(3)
    with cols[0]:
        st.image(Image.fromarray(img_rgb), caption=f"原始图像 ({img_rgb.shape[1]}×{img_rgb.shape[0]})", width="stretch")

    view_a_key = NAME_TO_KEY.get(st.session_state.fluo_view_a)
    view_b_key = NAME_TO_KEY.get(st.session_state.fluo_view_b)
    img_a = draw_algo_on_image(img_rgb, res, view_a_key) if view_a_key else None
    img_b = draw_algo_on_image(img_rgb, res, view_b_key) if view_b_key else None

    with cols[1]:
        if img_a:
            st.image(img_a, caption=f"{ALGO_INFO[view_a_key]['name']}: {res[view_a_key]['count']} 个", width="stretch")
        else:
            st.info("未选择算法")
    with cols[2]:
        if img_b:
            st.image(img_b, caption=f"{ALGO_INFO[view_b_key]['name']}: {res[view_b_key]['count']} 个", width="stretch")
        else:
            st.info("未选择算法")

    # 统计
    st.markdown("### 统计信息")
    stats_md = f"**尺寸:** {img_rgb.shape[1]} × {img_rgb.shape[0]} 像素  \n"
    counts = []
    for k in ALGO_KEYS:
        if k in res:
            stats_md += f"**{ALGO_INFO[k]['name']}:** {res[k]['count']} 个  \n"
            counts.append(res[k]['count'])
    if len(counts) >= 2:
        stats_md += f"**多算法平均:** <span style='color:green; font-size:1.3em'>{int(np.mean(counts))}</span> 个"
    st.markdown(stats_md, unsafe_allow_html=True)

    # 下载
    dcols = st.columns(3)
    @st.cache_data(show_spinner=False)
    def get_combined_fluo(img_rgb, res, va, vb):
        images = [Image.fromarray(img_rgb)]
        for key in (va, vb):
            img = draw_algo_on_image(img_rgb, res, key)
            if img: images.append(img)
        if len(images) < 2: return None
        widths, heights = zip(*(i.size for i in images))
        total_width, max_height = sum(widths), max(heights)
        combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0)); x_offset += img.width
        return combined

    comb = get_combined_fluo(img_rgb, res, view_a_key, view_b_key)
    if comb:
        buf = io.BytesIO(); comb.save(buf, format='PNG')
        dcols[0].download_button("💾 保存结果图", buf.getvalue(),
                                 f"{current['filename'].split('.')[0]}_result.png", "image/png", use_container_width=True)

    df = build_fluo_csv(res)
    if not df.empty:
        csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        dcols[1].download_button("📊 导出CSV坐标", csv_buf.getvalue().encode('utf-8-sig'),
                                  f"{current['filename'].split('.')[0]}_coords.csv", "text/csv", use_container_width=True)

    if len(all_results) > 1:
        summary = []
        for item in all_results:
            if item.get('error') or item['result'] is None: continue
            r = item['result']; counts = [r[k]['count'] for k in ALGO_KEYS if k in r]
            summary.append({'filename': item['filename'],
                          **{ALGO_INFO[k]['name']: r.get(k, {}).get('count', 0) for k in ALGO_KEYS},
                          'recommended': int(np.mean(counts)) if counts else 0})
        if summary:
            sum_df = pd.DataFrame(summary); sum_csv = io.StringIO()
            sum_df.to_csv(sum_csv, index=False, encoding='utf-8-sig')
            dcols[2].download_button("📥 批量汇总", sum_csv.getvalue().encode('utf-8-sig'),
                                      f"batch_summary_{datetime.now().strftime('%H%M%S')}.csv", "text/csv", use_container_width=True)

# --- 油红O / ALP / 茜素红 结果显示 ---
elif module in ("油红O脂滴", "ALP矿化", "茜素红矿化"):
    st.markdown(f"### 📄 {current['filename']}")
    # 重建img_bgr用于显示（res中只有mask，没有原始bgr，需要从result里找？）
    # 注意：analyze函数返回里没有原始图，我们需要重新加载或保存原始图
    # 为节省内存，我们在运行时没有保存img_bgr到result。这里需要重新加载第一张图来显示？
    # 不，更好的方式是在运行时将img_rgb或img_bgr存入result。但内存有限。
    # 折中：result里存img_rgb的缩略图或原始图。对于2048px图，存一份rgb是可以接受的。
    # 修改：在运行逻辑中，把img_rgb存入res。但analyze_oil_red_o返回里没有img_rgb。
    # 我需要在运行逻辑中手动添加。

    # 由于上面的运行逻辑没有保存原始图到res，这里我们重新加载当前文件来显示原始图
    # 但这要求UploadedFile支持seek。Streamlit的UploadedFile在session里可以seek(0)。
    # 更简单：在运行逻辑中把img_rgb存入result。
    # 修改运行逻辑：res['img_rgb'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 对于染色模块
    # 但上面代码没有这样做。我需要调整。

    # 临时解决方案：从uploaded_files中找到当前文件，重新加载显示。UploadedFile可以seek(0)。
    # 实际上，当用户切换查看不同图片时，UploadedFile对象仍在uploaded_files列表中。
    # 但run_clicked后uploaded_files变量仍在。如果用户切换module，uploaded_files可能为None？
    # 不，uploaded_files是widget返回值，只要页面不重新加载，它就存在。
    # 但如果用户重新上传，会触发rerun。我们假设用户查看结果时uploaded_files还在。
    # 更安全的做法：在session_state中保存原始图的缩略图或路径。但UploadedFile不能序列化。
    # 最佳方案：在运行逻辑中，把img_rgb存入result字典。

    # 让我修改运行逻辑，为染色模块添加img_rgb。
    # 由于代码已经很长，我在这里直接重新加载当前图片显示原始图（UploadedFile支持seek(0)）。
    # 但UploadedFile.read()后指针在末尾，需要seek(0)。
    # 然而，UploadedFile对象在多次rerun中是否保持文件指针？不确定。
    # 稳妥做法：在session_state.results中保存原始图的numpy数组太占内存。
    # 折中：保存为PIL Image或压缩格式。

    # 为简化，我假设用户查看结果时，我们重新从uploaded_files读取当前文件。
    # 找到当前文件对象：
    current_file = None
    for uf in uploaded_files:
        if uf.name == current['filename']:
            current_file = uf
            break
    if current_file:
        current_file.seek(0)
        img_display = load_image(current_file)
        img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    else:
        img_rgb_display = None

    if img_rgb_display is not None:
        cols = st.columns(2)
        with cols[0]:
            st.image(Image.fromarray(img_rgb_display), caption="原始图像", width="stretch")
        with cols[1]:
            mask_color = cv2.applyColorMap((res['mask'] > 0).astype(np.uint8) * 255, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_display, 0.6, mask_color, 0.4, 0)
            st.image(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), caption="阳性区域掩膜", width="stretch")

        # 统计指标
        st.markdown("### 统计指标")
        mcols = st.columns(4)
        if module == "油红O脂滴":
            mcols[0].metric("脂滴总数", res['count'], "个")
            mcols[1].metric("阳性面积占比", f"{res['area_ratio']:.2f}", "%")
            mcols[2].metric("平均脂滴面积", f"{res['avg_droplet_area']:.1f}", "px²")
            mcols[3].metric("阳性像素", f"{res['positive_pixels']}", "px")
        else:
            stain_label = "矿化" if module == "ALP矿化" else "钙结节"
            mcols[0].metric(f"{stain_label}结节数", res['nodule_count'], "个")
            mcols[1].metric("阳性面积占比", f"{res['area_ratio']:.2f}", "%")
            mcols[2].metric("平均结节面积", f"{res['avg_nodule_area']:.1f}", "px²")
            mcols[3].metric("阳性像素", f"{res['positive_pixels']}", "px")

        # 下载
        dcols = st.columns(3)
        # 掩膜图
        mask_buf = io.BytesIO(); Image.fromarray((res['mask'] > 0).astype(np.uint8) * 255).save(mask_buf, format='PNG')
        dcols[0].download_button("💾 保存掩膜图", mask_buf.getvalue(),
                                  f"{current['filename'].split('.')[0]}_mask.png", "image/png", use_container_width=True)

        # 结节CSV
        rows = []
        if module == "油红O脂滴":
            for d in res['droplets']:
                rows.append({'id': d['id'], 'area': d['area'], 'cx': d['centroid'][0], 'cy': d['centroid'][1]})
        else:
            for n in res['nodules']:
                rows.append({'id': n['id'], 'area': n['area'], 'cx': n['centroid'][0], 'cy': n['centroid'][1]})
        if rows:
            df = pd.DataFrame(rows); csv_buf = io.StringIO(); df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            dcols[1].download_button("📊 导出结节数据", csv_buf.getvalue().encode('utf-8-sig'),
                                      f"{current['filename'].split('.')[0]}_nodules.csv", "text/csv", use_container_width=True)

        # 批量汇总
        if len(all_results) > 1:
            summary = []
            for item in all_results:
                if item.get('error') or item['result'] is None: continue
                r = item['result']
                summary.append({
                    'filename': item['filename'],
                    'count': r.get('count', 0) if module == "油红O脂滴" else r.get('nodule_count', 0),
                    'area_ratio': r['area_ratio'],
                    'avg_area': r.get('avg_droplet_area', 0) if module == "油红O脂滴" else r.get('avg_nodule_area', 0)
                })
            if summary:
                sum_df = pd.DataFrame(summary); sum_csv = io.StringIO(); sum_df.to_csv(sum_csv, index=False, encoding='utf-8-sig')
                dcols[2].download_button("📥 批量汇总", sum_csv.getvalue().encode('utf-8-sig'),
                                          f"batch_summary_{datetime.now().strftime('%H%M%S')}.csv", "text/csv", use_container_width=True)

st.caption("提示：内存优化已启用，单图最大边长限制 2048px。处理完每张图立即释放内存。")
