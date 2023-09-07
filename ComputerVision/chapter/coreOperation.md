# 核心操作

# 1. 图像的操作

## 1.1. 图像的本质

- 图片是一个三维数组[像素高度,像素宽度,RGB值]
- 数组类型是 `uint8` 8位无符号整型

```python
#通过 numpy 自定义纯色图片：`blackImage = np.zeros(shape=(height,width,3),dtype=np.uint8)`

# 裁剪图片：将原图片的高度 100 - 200 的像素；宽度 50 - 100 的像素。提取出来
img[ 100:200,50:100,: ]

# 高度，宽度，颜色通道数 如果是灰度图仅有高度，宽度
h,w,c = img.shape

# 图像的总像素数
print(img.size)

# 图像数据类型
print(img.dtype)

# RGB 通道的拆分：结果为：高度像素 x 宽度像素 的二维数组
b,g,r = cv2.split(img)
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
# 合并多个被拆分出来的通道：将三个二维数组，组合成三维的数组
img = cv2.merge((b,g,r)) 

# 单通道图片
b = img[:,:,0]
g = img[:,:,1] * 0
r = img[:,:,2] * 0
imgB = cv2.merge((b,g,r)) 
```

## 1.2. 制作图像边界（填充）


- top,bottom,left,right-上下左右四个方向上的边界拓宽的值
- borderType-定义要添加的边框类型的标志。它可以是以下类型：
    > cv2.BORDER_CONSTANT - 添加一个恒定的彩色边框。该值应作为下一个参数value给出   
    > cv2.BORDER_REFLECT -边框将是边框元素的镜像反射，如下所示：fedcba|abcdefgh|hgfedcb   
    > cv2.BORDER_REFLECT_101或者 CV.BORDER_DEFAULT -与上面相同，但略有改动，如下所示： gfedcb | abcdefgh | gfedcba    
    > cv2.BORDER_REPLICATE -最后一个元素被复制，如下所示： aaaaaa | abcdefgh | hhhhhhh    
    > cv2.BORDER_WRAP -不好解释，它看起来像这样： cdefgh | abcdefgh | abcdefg    

```python
# copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst
cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)
```

# 2. 图像的算术运算

## 2.1. 加法
```python 
 # 图像直接相加 图像高宽通道必须一样
    imgA + imgB
    cv2.add(imgA,imgB)
    cv2.addWeighted(imgA, alpha, imgB, beta, gamma)
```

- `imgA + imgB`：**当数值大于一个字节时，大于一个字节的位数都被丢失了**。
$$
(A + B)  \%  256
$$

- `cv2.add(imgA,imgB)`：**当数值超过`255`时，取值为`255`**
$$
min(A+B,255)
$$

- `cv2.addWeighted(imgA, alpha, imgB, beta, gamma)`：
$$
min(round(A*alpha + B *beta + gamma),255) 
$$

## 2.2. 按位运算

```python
# 与运算
bitwise_and(src1:image, src2:image[, dst[, mask]]) -> dst
# 或运算
bitwise_or(src1:image, src2:image[, dst[, mask]]) -> dst
# 异或运算
bitwise_xor(src1:image, src2:image[, dst[, mask]]) -> dst
# 非
bitwise_not(src1:image[, dst[, mask]]) -> dst
```
- **与、或、异或：** 实质就是两个图像数组，相同位置的数据直接进行与、或、异或运算。
- **非：** 与程序中按位取反不一样，OpenCV 中实现的是对颜色反转

# 3. 仿射变换

## 3.1. 介绍 
**仿射变换**就是**线性变换**+**平移**

线性变换有三个要点：
- 变换前是直线的，变换后依然是直线
- 直线比例保持不变
- 变换前是原点的，变换后依然是原点（仿射变换没有这点）

## 3.2. 变换的数学表达

### 3.2.1 缩放
将原图的像素坐标进行缩放，其数学表达就为
$$
\begin{aligned}
x = \alpha_x x_c  \\
y = \alpha_y y_c  
\end{aligned}
$$
其矩阵形式为
$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 \\
  0 & \alpha_y
\end{bmatrix} 
\begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

### 3.2.2. 旋转

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/rotation_transform.jpg" width="50%" align="middle" /></p>

像素点$P_c(x_c,y_c)$ 点逆时针旋转 $\theta$ 角，旋转到 $P(x,y)$  位置。可以当作是坐标系 $Ox_c y_c$ 逆时针旋转 $- \theta$ 角后，$P_c$ 像素点在 $Oxy$ 坐标系的位置，因此可以推导出坐标变换
$$
\begin{aligned}
  x &= x_c \cos(-\theta)  + y_c \sin(- \theta)\\
  y &= - x_c \sin(-\theta) + y_c \cos(- \theta)
\end{aligned}
$$

化简得到矩阵形式

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) \\
  \sin(\theta) & \cos(\theta)
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

### 3.2.3. 平移

将原图的像素坐标进行移动，其数学表达就为
$$
\begin{aligned}
x = x_c + t_x \\
y = y_c + t_y 
\end{aligned}
$$

$(x_c,y_c)$ 为原图像素的坐标，$(x,y)$为变换后的像素坐标，$(t_x,t_y)$移动的距离。将上面公式改写成矩阵形式

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0 \\、
  0 & 1 
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix} + \begin{bmatrix}
  t_x \\
  t_y
\end{bmatrix}
$$

## 3.3. 变换矩阵

上一小节推导了平移、缩放、旋转的数学表达公式：

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0 \\
  0 & 1 
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix} + \begin{bmatrix}
  t_x \\
  t_y
\end{bmatrix}
$$


$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 \\
  0 & \alpha_y
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) \\
  \sin(\theta) & \cos(\theta)
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

从中可以看出缩放和旋转都能通过一个 2x2 的矩阵进行变换，而平移带有一个偏移项，为统一变换矩阵形式，就可以将平移改写为：

$$
\begin{bmatrix}
  x \\
  y \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0  & t_x \\
  0 & 1  & t_y \\
  0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\ 
  1
\end{bmatrix} 
$$

同样，也将旋转和缩放改写为 3x3 的形式：

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 & 0 \\
  0 & \alpha_y & 0 \\
  0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) & 0\\
  \sin(\theta) & \cos(\theta) & 0 \\
  0 & 0 & 1  
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

## 3.4. 变换矩阵的逆推

观察图中所有变换矩阵，其形式可以写为：
$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  0 & 0 & 1  
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**也就是说从图片A变换到图片B的「仿射变换矩阵」一共有`6`个未知数，而一组$(x,y),(x_c,y_c)$ 就能构造`2`组方程，所以需要`3`组对应的像素点坐标。**

## 3.5. 代码

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('图片')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
# 仿射变换矩阵逆推
# getAffineTransform(src, dst) -> retval
# src,dst : 三个顶点
M = cv2.getAffineTransform(pts1,pts2)

# 逆时针旋转转变换矩阵
# getRotationMatrix2D(center, angle, scale) -> retval
# # center，旋转中心
# angle，逆时针旋转角度
# scale，图片缩放值
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
# 仿射变换
# warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
# M:仿射变换矩阵  
# dsize :输出图片的大小  
# flags：图片的插值算法，默认算法INTER_LINEAR
# borderMode：查看 图像边界扩展
dst = cv2.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```


# 4. 透视变换

## 4.1. 齐次坐标

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  0 & 0 & 1  
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

仿射变换中，用来表示「二维像素位置」的坐标为

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix}
$$

从形式上来说，这就是用了「三维坐标」来表示「二维坐标」，即 **降维打击**。在将`1`进行符号化，用`w`进行代替

$$
\begin{bmatrix}
  x \\
  y \\
  w
\end{bmatrix}
$$

**这种表达 `n-1` 维坐标的 `n` 维坐标，就被称之为「齐次坐标」。**

## 4.2. 透视

透视的目的就是实现 **近大远小**，也就是需要有纵向的深度，而像素位置 $(x,y)$ 只能表示像素在平面上的位置关系，此时「齐次坐标」就能排上用场了。三维的齐次坐标虽然表示的二维的平面，但是其本质还是一个三维空间的坐标值，这样就能将图片像素由「二维空间」扩展到「三维空间」进行处理，齐次坐标的`w`分量也就有了新的含义：三维空间的深度。

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/perspective.jpg" width="50%" align="middle" /></p>


在「仿射变换」中，像素的齐次坐标为 $[x,y,1]^T$，可以解释为图像位于三维空间 的 $w=1$ 平面上，即 $w=1$ 平面就是我们在三维空间中的视线平面（三维空间中的所有东西都被投影到 $w=1$ 平面，然后我们才能看见）。「透视」就规定了所有物体如何投影到视线平面上，即「近大远小」。数学描述就是根据像素三维空间中的坐标点 $P(x,y,w)$ 得出像素在视线平面上的坐标 $P_e(x_e,y_e,1)$，两个关系如图所示，根据三角形相似定理就能得出：

$$
\begin{aligned}
  \frac{x}{x_e} = \frac{w}{1} \\
  \frac{y}{y_e} = \frac{w}{1}
\end{aligned}
$$

整理得：

$$
\begin{aligned}
  x_e = \frac{x}{w} \\
  y_e = \frac{y}{w} \\
  1 = \frac{w}{w}
\end{aligned}
$$

上述公式就实现了三维空间像素坐标向视线平面的透视投影。

## 4.3. 透视变换

$$
\begin{bmatrix}
  x \\
  y \\
  w 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

根据「仿射变换」可知，上述矩阵就能实现图片像素坐标 $[x_e,y_e,1]^T$ 在三维空间中的旋转、缩放、切变的变换操作（没有三维空间的平移，变换矩阵差一个维度），得到像素位置变换后的三维坐标就为 $[x,y,w]^T$。再将新的像素齐次坐标进行透视处理，将坐标映射到 $w=1$ 平面， 得到的像素位置就是最终「透视变换」的结果。

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/perspectiveTransform.jpg" width="50%" align="middle" /></p>

因此透视变换的变换矩阵就能改写为

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \frac{1}{w}
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

由于`w`是一个常量，也可以放入变换矩阵：
$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

将矩阵拆解

$$
\begin{aligned}
x' &= a_{11} x_c + a_{12} y_c + a_{13}\\
y' &= a_{21} x_c + a_{22} y_c + a_{23} \\
1 &= a_{31} x_c + a_{32} y_c + a_{33} \\
\end{aligned}
$$

根据齐次坐标透视规则

$$
\begin{aligned}
x' &= \frac{a_{11} x_c + a_{12} y_c + a_{13}}{a_{31} x_c + a_{32} y_c + a_{33}} \\
y' &= \frac{a_{21} x_c + a_{22} y_c + a_{23}}{a_{31} x_c + a_{32} y_c + a_{33}}  
\end{aligned}
$$

可以看出，对分式上下乘以一个非零常数 $\alpha$ ，值不变
$$
\begin{aligned}
x' &= \frac{ \alpha ( a_{11} x_c + a_{12} y_c + a_{13} )}{\alpha( a_{31} x_c + a_{32} y_c + a_{33} )} \\
y' &= \frac{\alpha( a_{21} x_c + a_{22} y_c + a_{23} )}{\alpha( a_{31} x_c + a_{32} y_c + a_{33} )}  
\end{aligned}
$$

再将上面的式子写成矩阵形式

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \alpha
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

可以看出变换矩阵乘以一个非零常数，对结果无影响，那么就直接令 $\alpha = \frac{1}{a_{33}}$

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \frac{1}{a_{33}}
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$


其结果就为

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & 1\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**可以看出两张图片之间的透视变换，只涉及`8`个自由度。**


> [!tip|style:flat]
> 从最后的公式形式可以看出，仿射变换其实就是透视变换的一种特例，仿射变换只是 $w=1$ 的平面内进行平移、缩放、旋转等。

## 4.4. 透视变换逆推

根据下式可知，一对像素坐标点 $(x_c,y_c),(x',y')$ 只能构成`2`组方程
$$
\begin{aligned}
x' &= \frac{a_{11} x_c + a_{12} y_c + a_{13}}{a_{31} x_c + a_{32} y_c + a_{33}} \\
y' &= \frac{a_{21} x_c + a_{22} y_c + a_{23}}{a_{31} x_c + a_{32} y_c + a_{33}}  
\end{aligned}
$$
然后透视变换矩阵具有`8`个未知变量，所以**逆向求解变换矩阵需要`4`对像素坐标**。
$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & 1\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

## 4.5. 代码

```python 

rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

## 逆向计算透视变换矩阵
# getPerspectiveTransform(src, dst[, solveMethod]) -> retval
M = cv2.getPerspectiveTransform(pts1,pts2)

## 透视变换
# warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
# M ：透视变换矩阵 3x3
# dsize : 要显示的图片大小
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

# 5. 图像边界扩展

**作用：** 当图像需要变大，但是不想直接缩放，则可以选择不同的方法将图片外围进行扩展，使得原图变大

```python
destImage = cv2.copyMakeBorder(src: Mat, 
                              top_size, bottom_size, left_size, right_size, 
                              borderType)
```
- `top_size, bottom_size, left_size, right_size`：图片上下左右，需要填充的像素值
- `borderType`：填充方式
  - `BORDER_REPLICATE`:复制法，将最边缘像素向外复制。
  - `BORDER_REFLECT`:反射法，对感兴趣的图像中的像素在两边进行复制例如：dcba | abcd（原图） | dcba
  - `BORDER_REFLECT1O1`:反射法，也就是以最边缘像素为轴，例如 dcb | abcd（原图） | cba，**没有复制最边缘的像素**
  - `BORDER_WRAP`:外包装法，abcd | abcd | abcd ，重复原图
  - `BORDER_CONSTANT`:常量法，边界用常数值填充。111 | abcd | 111

  <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/makeBorder_categories.jpg" width="50%" align="middle" /></p>