# 目标识别
# 1. Opencv 实现

## 1.1. 人脸检测

> [!tip]
> haar 算法计算速度快，但是精度相对较低。   


- **作用：** <span style="color:red;font-weight:bold"> 从图像中找出人脸的位置。 </span> 

<!-- panels:start -->
<!-- div:left-panel -->

- <a href="https://www.cnblogs.com/wumh7/p/9403873.html" class="jump_link"> haar 算法实现 </a>：
  1. 提取类Haar特征：
  2. 利用积分图法对类Haar特征提取进行加速；
  3. 使用Adaboost算法训练强分类器，根据特征区分出人脸和非人脸；
  4. 使用筛选式级联把强的分类器级联在一起，从而提高检测准确度。

<!-- div:right-panel -->
<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/haarClassify.jpg" width="50%" align="middle" /></p>
<!-- panels:end -->


- **haar 特征提取：** 将 Haar-like 特征在图片上进行滑动，在每个位置计算白色区域对应的像素值的和减去黑色区域对应的像素值的和，从而提取出该位置的特征，
    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/haar_like.jpg" width="50%" align="middle" /></p>

- <a href="https://github.com/opencv/opencv/tree/4.x/data/haarcascades" class="jump_link">OpenCV 提供的模型  </a> 或依照路径在源码中找

```python

# 加载模型
detector = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# scaleFactor：两个相邻窗口的间隔比例
# minNeighbors：弱分类器要满足多少个，才认为是目标
# flags：兼容旧版
# minSize：目标对象可能的最小尺寸
# maxSize：目标对象可能的最大尺寸
# objects：所有目标的 x,y,w,h
# cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize:tuple [, maxSize: tuple]]]]] -> objects
faces = detector.detectMultiScale(frameGray,1.2,3,0)

```

<details>
<summary><span class="details-title">完整代码</span></summary>
<div class="details-content"> 

```python
import cv2
import numpy as np

# 加载级联分类器模型文件
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取视频帧
    ret, frame = cap.read()

    if not ret:
        break

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用级联分类器检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 在帧上绘制检测到的人脸矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('Face Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1000//30) & 0xFF == ord('q'):
        break

# 释放视频捕捉器和关闭窗口
cap.release()
cv2.destroyAllWindows()

```

</div>
</details>

## 1.2. 人脸识别

**OpenCV 实现太繁琐，方法太旧，懒得学了 (￣_,￣ ) ， dlib 要更靠谱一些。**

- **作用：**  <span style="color:red;font-weight:bold">  对人脸特征进行对比，区分出谁是谁。 </span>

- **算法：**  
    - <a href="https://cloud.tencent.com/developer/article/1082468" class="jump_link"> EigenFace </a>
    - FisherFace
    - LBPH

- **PCA 主成分分析：** EigenFace 算法会利用到的一种算法。**通过PCA方法可以对原始数据进行降维处理，重点对特征分量进行分析。**
    - 使得数据集更易使用。
    - 降低算法的计算开销。
    - 去除噪声。
    - 使得结果容易理解。

    <p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/PCA.jpg" width="100%" align="middle" /></p>


# 2. dlib 实现

## 2.1. dlib 安装

> - <a href="http://dlib.net/" class="jump_link" target="_blank"> dlib 官网 </a>
> - <a href="https://github.com/sachadee/Dlib" class="jump_link" target="_blank"> python 3.7 ~ 3.9 编译好版本 </a>
> - <a href="https://pypi.org/simple/dlib/" class="jump_link" target="_blank"> python 3.4 ~ 3.6 编译好版本 </a>
> - <a href="https://www.bilibili.com/video/BV1Ht4y1U7Do" class="jump_link" target="_blank"> 自己编译 </a>

## 2.2. 人脸检测

人脸检测的内部实现靠的就是 HOG 描述符、SVM 等算法实现。

```python
# 获取默认的检测
detector = dlib.get_frontal_face_detector()

# upsample_num_times ：对图片进行上采样（放大）多少次
# return：rectangles
# 对图片进行检测
faces = detector(image: array, upsample_num_times: int=0L)

# rectangle 类
y1 = rectangle.bottom()  # detect box bottom y value
y2 = rectangle.top()  # top y value
x1 = rectangle.left()  # left x value
x2 = rectangle.right()  # right x value
```

<details>
<summary><span class="details-title">完整代码</span></summary>
<div class="details-content"> 

```python
import cv2
import dlib

# 获取默认的检测
detector = dlib.get_frontal_face_detector()

video = cv2.VideoCapture(0)

while video.isOpened():
    # 读取
    flag,frame = video.read()
    if flag == False:
        break

    frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frameGray,0)

    # 标记
    for face in faces:
        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('face',frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

```

</div>
</details>


## 2.3. 人脸追踪

上述的人脸检测步骤其实只适用于「单张图片」的人脸检测，如果对视频的每一帧都使用同样的方法一帧图片一帧图片检测，在 dlib 中可能会很慢。为了加快视频中的人脸检测，可以采用追踪的方式。

```python
# 获取默认的检测
detector = dlib.get_frontal_face_detector()

# 追踪器
tracker = dlib.correlation_tracker()

# 定位人脸
face:dlib.rectangle = detector(frame,0)[0]

# 启动追踪
tracker.start_track(frame,face)

# 更新追踪
tracker.update(frame)
# NOTE -  追踪的结果为浮点数，需要转为整型
face:dlib.drectangle = tracker.get_position()
```

> [!tip|style:flat]
> - 追踪器其实只要初始化时，给定一个 `dlib.rectangle`位置，之后就会追踪这个区域，因此，只要初始化时，给定一个目标位置，追踪器就能够对目标进行追踪，不一定非得是人脸。
> - 当被追踪的目标跑出画面后，然后又跑回来，追踪器就可能追踪不了了。

## 2.4. 人脸特征位置

- <a href="http://dlib.net/files/" class="jump_link"> dlib 人脸关键点预测模型 </a>

- <a href="https://blog.csdn.net/YeziTong/article/details/86177846" class="jump_link"> 具体实现 </a>


**获取特征点位置：**

```python
# 加载关键点预测器
predictor:dlib.shape_predictor = dlib.shape_predictor('./asset/shape_predictor_68_face_landmarks.dat')

# 预测关键点
points: dlib.full_object_detection = predictor(img,face)

# 遍历点
for i in range(len(points.parts())):
    point:dlib.point = points.part(i)
    # x 坐标
    point.x
    # y 坐标
    point.y
```

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/dlib_facePoints.png" width="50%" align="middle" /></p>

**对于 dlib 通过模型找出的人脸特征点，输出结果是具有顺序的。通过对应位置的特征点，我们就能标记出眼睛、鼻子、嘴巴、眉毛的位置。** <span style="color:red;font-weight:bold"> 图上特征点的索引是从`1`开始的，在编程的时候，数组索引是从`0`开始的。 </span>

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/dlib_pointNumber.png" width="50%" align="middle" /></p>

**特征点连线：**

```python
# 转化点的类型
pts = np.array([( point.x,point.y )for point in points.parts()],dtype=np.int32)

# 左眼提取点全部连起来
cv2.polylines(img, [pts[36:42]], True,(255,0,0),2)
```

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/dlib_eye.png" width="50%" align="middle" /></p>

## 2.5. 人脸识别

> - <a href="http://dlib.net/files/" class="jump_link" target="_blank"> 残差神经网络模型 </a>

**实现步骤：**
1. 检测出人脸位置
2. 预测出人脸的特征点位置
3. 将特征点通过残差神经网络转化为`128`维的特征描述符
4. 对比两张人脸图片的特征描述符（最简单的方法就是计算欧式距离），就能确定两个图片是否为同一个人

```python
# 加载残差神经网络模型
encoder = dlib.face_recognition_model_v1('./asset/dlib_face_recognition_resnet_model_v1.dat')

# 生成 128 维的特征描述符 
description = encoder.compute_face_descriptor(img,keypointsLoc,jet)
```

<details>
<summary><span class="details-title">案例代码</span></summary>
<div class="details-content"> 

```python
import dlib
import numpy as np
import  cv2

def preprocess(path,fx=0.5,fy=0.5):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0),fx=fx,fy=fy)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img,imgGray)

def imshow(img,title='untitled'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def lableFaces(canvas,facesLocs):
    for face in facesLocs:
        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),2)

def facesKeypointDescritptions(img,imgGray,facesLocs,predictor,encoder,jet=1):
    # 特征点位置
    keypointsLocs = [predictor(img,faceLoc) for faceLoc in facesLocs]

    # 获取描述符
    return np.array([encoder.compute_face_descriptor(img,keypointsLoc,jet) for keypointsLoc in keypointsLocs])

if __name__ == '__main__':
    # 载入图片
    facesImg,facesImgGray = preprocess('./asset/faces.jpg')
    targetImg,targetImgGray = preprocess('./asset/mads.png')

    # 人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 特征点预测器
    predictor = dlib.shape_predictor('./asset/shape_predictor_68_face_landmarks.dat')

    # 特征描述生成模型
    encoder = dlib.face_recognition_model_v1('./asset/dlib_face_recognition_resnet_model_v1.dat')
    
    #  标定人脸位置
    facesLocs = detector(facesImgGray,0)
    targetLocs = detector(targetImgGray,0)

    # 获取人脸特征描述
    facesDescriptions = facesKeypointDescritptions(facesImg,facesImgGray, facesLocs, predictor, encoder)
    targetDescription = facesKeypointDescritptions(targetImg,targetImgGray, targetLocs, predictor, encoder)

    print(facesDescriptions.shape)
    print(targetDescription.shape)

    # 描述符对比，计算欧氏距离
    distances = np.linalg.norm(facesDescriptions - targetDescription,axis=1)

    print(np.argmin(distances))

    # 将结果标记出来
    lableFaces(facesImg, [facesLocs[np.argmin(distances)]])

    imshow(facesImg)
    imshow(targetImg)
``` 

</div>
</details>

<p style="text-align:center;"><img src="https://cdn.jsdelivr.net/gh/chenlfdev/pic-store/img/dlib_faceRecognition.png" width="75%" align="middle" /></p>
