# 基础操作

# 1. OpenCV安装
1. 安装 [Anaconda](https://www.anaconda.com/) [Anaconda国内清华镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)  [源](https://pypi.tuna.tsinghua.edu.cn/simple/opencv-python/)
2.  `Anconda` 创建环境
 ```term
conda create --name 新环境名 python=版本号
conda activate 环境名 // 启动环境
conda deactivate  // 退出当前环境
conda install  opencv-python opencv-contrib-python // 直接装最新的 opencv
```

# 2. OpenCV图像

## 2.1. OpenCV窗口
```python
# 导入 OpenCV
import cv2 


#第一个参数是窗口名，第二个参数是图像，多个窗口窗口名要不同
cv2.imshow('image', img)

# 等待按键
# 1. 将窗口堵塞。等带按键、并会返回按键的ASCII码
# 2. 可以设置等待的时间，单位 ms
# 3. 等待时间为 0 ，则为一直等待
cv2.waitKey(0)
#销毁所有窗口 销毁指定窗口用 cv2.destroyWindows(‘窗口名’)
cv2.destroyAllWindows()
```

有种情况可以创建窗口再加载图像到窗口。由函数`cv2.namedWindow()`完成，能指定窗口是否可调整大小。flag默认是`cv2.WINDOW_NORMAL`,`cv2.WINDOW_NORMAL`才能设置窗口大小
```python
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2.2. OpenCV图像

```python
#读取图片
# flag 指定了读取图像的方式
# cv2.IMREAD_COLOR  加载彩色图像，它是默认标志 BGR模式
# cv2.IMREAD_GRAYSCALE 以灰度模式加载图像
# cv2.IMREAD_UNCHANGED 加载图像，包括 alpha 通道
# 或者用整数 1、0 或-1
img=cv2.imread('图片路径',flag)

# 图片保存
cv2.imwrite('保存路径',img)
```



# 3. OpenCV视频

## 3.1. 视频读取
获取一个视频需要创建一个`VideoCapture`对象。它的参数可以是**设备索引**或者一个**视频文件名**。设备索引是相机编号。

```python
import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened(): #检查是否初始化，没有用cap.open()打开
    # 一帧一帧捕捉
    ret, frame = cap.read()
    if ret == True:
        # 我们对帧的操作在这里 灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 显示返回的每帧
        cv2.imshow('frame',gray)
    #控制播放速度 以 60 帧的速度进行图片显示
    if cv2.waitKey(1000 // 60) == ord('q'):
        break
# 当所有事完成，释放 VideoCapture 对象
cap.release()
cv2.destroyAllWindows()
```

## 3.2. 视频写入

创建一个 `VideoWriter` 对象。我们应该指定输出文件的名字 (例如：output.mp4)。然后我们应该指定 `FourCC` 码 。然后应该传递每秒帧数和帧大小。最后一个是 isColor flag。如果是 True，编码器期望彩色帧，否则它适用于灰度帧。

FourCC 是用于指定视频解码器的 4 字节代码。这里 fourcc.org 是可用编码的列表。它取决于平台，下面编码就很好。
- In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID 是最合适的. MJPG 结果比较大. X264 结果比较小)
- In Windows: DIVX (还需要测试和添加跟多内容)
- In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).
```python
import cv2
cap = cv2.VideoCapture(0)
# 声明编码器和创建 VideoWrite 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 用录制视频的尺寸，不然保存后有可能打不开
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 保存路径，保存格式，保存的视频帧数，（宽度像素，高度像素） 
out = cv2.VideoWriter('output.mp4',fourcc, 24, (width,height)) #视频输出帧数
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
         cv2.imshow('frame',frame)
        # 保存
        out.write(frame)
        if cv2.waitKey(1000//24) == ord('q'): #摄像头拍照帧数
            break
    else:
        break
# 释放已经完成的工作
cap.release()
out.release()
cv2.destroyAllWindows()
```

# 4. 回调函数

```python
import cv2
import numpy as np
# 定义鼠标回调函数
# event：事件类型
# x,y：鼠标所在像素值
# userdata：用户传入数据
def mouse_callback(event,x,y,flags,userdata:any):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event,x,y,flags,userdata)
# 创建窗口
cv2.namedWindow('Event Test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Event Test',width=640,height=380)
# 鼠标事件指定回调函数
cv2.setMouseCallback('Event Test',mouse_callback,"userdata")
# 生成一个背景图片
bg = np.zeros((380,640,3),dtype=np.uint8)
while True:
    cv2.imshow('Event Test',bg)
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()
```