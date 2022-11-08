##  Real-time Infrared Small Target Segmentation based on MobileNetv3-UNet（C++ & Libtorch version）
## 基于MobileNetV3-Unet的实时红外目标分割（C++ & libtorch部署版本）

 This project is an Unet version deployed using libtorch. It can reach 20FPS per second on CPU (R5-5600H), and can achieve real-time with GPU acceleration.


## Dependence

    cmake
    gcc 5.4 +


## Demo
```
mkdir build && cd build
cmake ..
```

```
make
```

```
unet_libtorch
```

![img]("./images/from_raw.png")     ![mask]("./results/result.jpg") 
