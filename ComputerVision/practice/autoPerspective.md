# 自动透视变换

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/perspectivePaper.jpg" width="25%" align="middle" /></p>

- **目的：** 将图片中的纸张通过透视变换变正
- **思路：** 需要先将纸张的四个角点找出，然后计算透视变换矩阵，最后透视变换。
- **实现：**
    1. 图片预处理：缩放、去噪、转灰度等
    2. 获取纸张边缘：由于是查找纸张的外轮廓，并关心其他细节，因此采用边缘检测算法更合理一些
    3. 检索轮廓：对纸张边缘进行轮廓检索，轮廓面积最大的就是纸张的轮廓
    4. **提取角点：** 对纸张轮廓进行「轮廓近似」，由于四边形的四条边接近于直线，这样轮廓近似后，所保留下的轮廓点就是四个角点。
    5. 进行透视变换

- <a href="https://github.com/chenlfdev/chenlfdev.github.io/tree/main/example/ComputerVision/autoPerspective" class="jump_link" target="_blank"> 案例项目工程 </a>