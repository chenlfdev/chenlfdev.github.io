# 绘图功能

# 1. 图形

## 1.1. 直线、圆形、矩形

```python
import cv2
import numpy as np

# 创建一个黑色的图像
img = np.zeros((512,512,3), np.uint8)
# 直线
# 画一条 5px 宽的对角线
cv2.line(img,(0,0),(511,511),(255,0,0),5)

# 圆形
# circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
cv2.circle(img,(200,60),50,(0,0,255),3,16)

# rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
cv2.rectangle(img,(20,40),(100,100),(255,0,0),3)

cv2.imshow('draw_shape',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 1.2. 椭圆
- center：圆心
- axes：长轴、短轴 
- angle：椭圆倾斜角度，顺时针
- startAngle:椭圆弧的起始角度
- endAngle:椭圆弧的终止角度
```python
#  ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img
cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)
```

## 1.3. 多边形
- pts： 坐标点
- isClosed：为False 不会闭合线条 
```python
pts1 = np.array([ (20,60),(300,150),(50,300) ])
pts2 = np.array([ (400,60),(300,100) ])
# polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> img
cv2.polylines(img,[pts1,pts2],True,(0,255,255))
```
# 2. 文本

- text：文本字符串
- org：坐标（x,y）
- fontFace：字体
- fontScale：字体比例
```python
# putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
cv2.putText(img,"opencv",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5)
```

# 附录: API传参注释

```python
circle(canvas:img, center:tuple:, radius, 
    color[, thickness[, lineType[, shift]]]) -> img
```

在API中，`[]`代表了可选参数，函数调用时，这些参数不用设置。因此上面的函数必须传递的参数是：`canvas:img, center:tuple:, radius, color`

此外，`[]`还可以嵌套使用，例如`[, thickness[, lineType[, shift]]]`，其含义就是：
1. 在选择传输参数`thickness`之后，`[, lineType[, shift]]` 部分为可选参数，
2. 必须在设置了参数`thickness`的前提下，才能设置参数`[, lineType[, shift]]`