# Image_Processing_ShanHaiJing

本项目是一个以《山海经》为主题的叙事型图像处理系统，通过图像分割、空间约束与外观协调等方法，实现真实人像自然嵌入生成的神话世界，并生成定格画面与简短的展示视频。

## 运行方式

### 1. 安装依赖（项目根目录执行）
```bash
pip install -e .
```

### 2. 启动后端服务
在 backend 目录下执行：

```bash
uvicorn app:app --reload
```

### 3. 运行前端
直接使用浏览器打开 index.html 文件即可开始交互。

注：视频生成过程可能需要一定时间，请耐心等待。

权重文件过大，难以上传至github，完整版请见如下链接：

https://disk.pku.edu.cn/link/AA65D7F77BA78541AA836DCDBD386B77BC
