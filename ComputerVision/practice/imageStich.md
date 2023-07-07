# 图片拼接

# 1. 项目

- **思路**：
  1. 获取图片的关键点、描述符
  2. 得到两张图片的点匹配关系
  3. 通过 RANSAC算法计算透视变换矩阵：直接调用函数 `findHomography`
  4. 对目标图片进行透视变换，然后与另外一张图片合并

- <a href="https://github.com/fupobaobaowoya/fupobaobaowoya.github.io/tree/main/example/ComputerVision/imageStich" class="jump_link"> 案例项目工程 </a>

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/imageMerge.png" width="75%" align="middle" /></p>

- `ransacReprojThreshold`：当进行转换矩阵计算时，RANSAR 算法区分内点与外点的阈值。
    $$
    |X_{dst} - HX_{src}| > threshold
    $$

    $X_{dst}$ 与 $X_{src}$ 是一对匹配点；$H$ 为RANSAC算法得到的转换矩阵。当满足上述关系时，就认为 $X_{dst}$ 与 $X_{src}$  是异常匹配关系（外点）。



# 附录：RANSAC算法

## 介绍

- **解决问题：** 当利用 <a href="https://zhuanlan.zhihu.com/p/38128785" class="jump_link"> 最小二乘法 </a> 处理多样本拟合问题时，最小二乘会受到噪点的干扰。
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/lastSquareDisturbe.jpg" width="50%" align="middle" /></p>

- **RANSAC算法：** RANSAC通过反复选择数据中的一组随机子集来进行目的计算，例如最小二乘运算（RANSAC算法只利用了部分样本，而最小二乘是直接用全部样本）。重复多次后，选择结果最好的模型。

- **RANSAC作用：** 主要解决样本中的外点问题。
  - 外点：一般为数据中的噪声，比如说匹配中的误匹配和估计曲线中的离群点。

## 算法流程

1. 选择出可以估计出模型的最小样本集合：直线拟合来说就是两个点
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/RANSAC_sample.jpg" width="25%" align="middle" /></p>
2. 使用这个数据集来计算出数据模型：通过两点计算直线方程
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/RANSAC_model.jpg" width="25%" align="middle" /></p>
3. 将所有数据带入这个模型，计算出「内点」的数目：上述模型能够正确描述的点（绿色的）
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/RANSAC_innerPoints.jpg" width="25%" align="middle" /></p>
4. 重复上述步骤m次，然后从中选择出「内点」数组最多的模型
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/fupobaobaowoya/pic-store/img/RANSAC_good.jpg" width="25%" align="middle" /></p>

## 参数确定

内点占据样本比列

$$
p = \frac{n_{in}}{n_{in} + n_{out}}
$$

每次选择 k 个样本，选中一个外点的概率

$$
1 - p^k
$$

M 次重复样本都包含外点的概率，即模型计算错误的概率

$$
(1-p^k)^M
$$

样本都是内点的概率，即模型计算正确的概率

$$
z = 1 - (1-p^k)^M
$$

重复次数就为

$$
M = \frac{log(1 - z)}{log(1-p^k)}
$$