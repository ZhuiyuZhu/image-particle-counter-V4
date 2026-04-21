# image-particle-counter-V4
图片颗粒计数器，用于计数包括细胞在内的镜下小颗粒物，可选多种模式，适于大部分应用场景
# 🔬 荧光颗粒检测器 v4.1

基于 Python + Streamlit 的网页版荧光颗粒/细胞计数工具，支持 5 种检测算法、量化调参和批量处理。

## ✨ 功能特性

- **5 种检测算法**
  - **LoG 斑点检测**：标准拉普拉斯高斯，精度最高，提供半径估计
  - **DoG 快速检测**：高斯差分近似 LoG，速度快 2 倍，同样提供半径
  - **霍夫圆检测**：基于霍夫变换投票，对正圆颗粒/细胞极稳健
  - **局部最大值**：仅检测亮度峰值，速度最快，适合极小高密度颗粒
  - **自适应阈值+连通域**：应对背景不均、粘连团块和不规则形状- **量化调参系统**
  - 检测严格度（0=高召回 ~ 100=高精度）
  - 期望颗粒尺寸级别、尺寸均匀度、最小像素间距
  - 一键快速跳转：高召回 / 平衡 / 高精度

- **计数对象预设**
  - 通用荧光颗粒、细胞/大圆颗粒、小点状颗粒/病毒样颗粒、致密团块/细胞团
  - 切换预设自动加载推荐算法组合和参数

- **结果对比视图**
  - 左侧可勾选任意 N 种算法运行
  - 右侧从已勾选算法中任选其二并排对比显示
  - 支持结果图下载、CSV 坐标导出、批量汇总统计

- **内存优化**
  - 单图最大边长限制 2048px（防止 OOM）
  - float32 计算、num_sigma=5、处理完立即垃圾回收
  - 适配 Streamlit Cloud 免费版 1GB 内存限制
## 🚀 在线部署（Streamlit Cloud）

1. Fork 本仓库或新建 GitHub 仓库，包含以下文件：
   ├── streamlit_app.py
   ├── requirements.txt
   └── packages.txt # 可选，若报错则添加 libgl1
2. 访问 [share.streamlit.io](https://share.streamlit.io) 并连接仓库
3. 点击 **Deploy**，等待依赖安装完成即可访问

> 如果部署时遇到 `libGL.so.1` 错误，请在仓库根目录创建 `packages.txt`，内容如下：
  libgl1 libglib2.0-0
## 🖥️ 本地运行
# 克隆仓库
git clone https://github.com/yourusername/image-particle-counter.git
cd image-particle-counter

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动 Streamlit
streamlit run streamlit_app.py
浏览器将自动打开  http://localhost:8501 。
