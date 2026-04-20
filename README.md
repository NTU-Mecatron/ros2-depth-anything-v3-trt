# Depth Anything V3 TensorRT ROS 2 Node


https://github.com/user-attachments/assets/d119d3b8-bba1-43a3-9f86-75db24e01235


A ROS 2 node for Depth Anything V3 depth estimation using TensorRT for real-time inference. This node subscribes to camera image and camera info topics and publishes directly both, a metric depth image and `PointCloud2` point cloud.

<!-- omit from toc -->
## Overview
- [Features](#features)
- [Topics](#topics)
- [Parameters](#parameters)
- [Usage](#usage)
- [Architecture](#architecture)
- [Depth Postprocessing Pipeline](#depth-postprocessing-pipeline)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

Proceed to the [Usage](#usage) section for quick start instructions, or read through the detailed documentation below for in-depth information on configuration and troubleshooting.

## Features

- **Real-time metric depth estimation** using Depth Anything V3 with TensorRT acceleration
- **Point cloud generation** from metric depth image
- **Debug visualization** with colormap options
- **Configurable precision** (FP16/FP32)

> [!IMPORTANT]  
> This repository is open-sourced and maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/).  
> We cover a wide variety of research topics within our [*Vehicle Intelligence & Automated Driving*](https://www.ika.rwth-aachen.de/en/competences/fields-of-research/vehicle-intelligence-automated-driving.html) domain.  
> If you would like to learn more about how we can support your automated driving or robotics efforts, feel free to reach out to us!  
> :email: ***opensource@ika.rwth-aachen.de***

## Topics

### Subscribed Topics

- `image/raw` (sensor_msgs/msg/Image): Input RGB image topic
- `camera_info` (sensor_msgs/msg/CameraInfo): Input camera calibration topic

These are fixed relative topic names. Change them with ROS remapping, not parameters.

### Published Topics  

- `depth_image/raw` (sensor_msgs/msg/Image): Raw metric depth image topic relative to the node namespace
- `point_cloud` (sensor_msgs/msg/PointCloud2): Point cloud topic relative to the node namespace

## Parameters

Model and runtime parameters can be configured via `config/depth_anything_v3.param.yaml` or passed at launch time. Topic names are not parameters; use ROS remapping.

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `onnx_path` | string | `"models/DA3METRIC-LARGE.onnx"` | Path to Depth Anything V3 ONNX or TensorRT engine file |
| `precision` | string | `"fp16"` | Inference precision (`"fp16"` or `"fp32"`) |

### Sky Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sky_threshold` | double | `0.3` | Threshold for sky classification from model's sky output (lower = more sky detected) |
| `sky_depth_cap` | double | `200.0` | Maximum depth value (meters) to assign to sky pixels |
| `publish_compressed_depth` | bool | `false` | Publish `depth_image/raw` through `image_transport` so compressed depth transport plugins can be used |

### Point Cloud Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `publish_point_cloud` | bool | `true` | Whether to publish the point cloud output |
| `point_cloud_downsample_factor` | int | `2` | Publish every Nth point (1 = full resolution, 10 = every 10th point) |
| `colorize_point_cloud` | bool | `true` | Add RGB colors from input image to point cloud |

## Usage

#### Model Preparation

1. Download the ONNX file from Huggingface: [https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX](https://huggingface.co/TillBeemelmanns/Depth-Anything-V3-ONNX)
2. Place the ONNX/engine file in the `models/` directory
3. Update `config/depth_anything_v3.param.yaml` with the correct *ONNX path*
4. Build the package, followed by building the engine for the first time: `ros2 launch depth_anything_v3 depth_anything_v3.launch.py`
5. After this initial run, the TensorRT engine will be created. Update the `onnx_path` parameter to point to the generated engine file for faster subsequent launches.

#### Subsequent Launch

```bash
ros2 launch depth_anything_v3 depth_anything_v3.launch.py
```

#### With Custom Topics

Apply remappings where the component is loaded, for example in your bringup package. The node uses fixed relative topic names:

- `image/raw`
- `camera_info`
- `depth_image/raw`
- `point_cloud`

Use standard ROS remapping rules in the parent launch/container if those interfaces need different names.

#### Performance
The node is optimized for real-time performance.

Performance for DA3METRIC-LARGE:
- **Quadro RTX 6000**: 50 FPS
- **RTX 5070Ti**: 20 FPS
- **Jetson Orin NX**: 7.5 FPS

## Architecture

```
Input Image + Camera Info
         ↓
    Preprocessing (CPU/GPU)
         ↓  
    TensorRT Inference (GPU)
         ↓
    Postprocessing (CPU)
         ↓
   Depth Image + Point Cloud
```

## Depth Postprocessing Pipeline

Depth Anything V3 predicts a dense metric depth map. After TensorRT inference, the node performs the following postprocessing steps:

### 1. Depth Extraction
- The single-channel NCHW tensor is reshaped into a `cv::Mat`
- Negative depth values are clamped to zero

### 2. Focal Length Scaling
- The raw depth output is scaled based on the camera's focal length
- Scale factor: `focal_pixels / 300.0` where `focal_pixels = (fx + fy) / 2`
- This aligns the model's internal focal normalization with your actual camera intrinsics

### 3. Sky Handling
The model outputs a separate sky classification tensor:
- Pixels with sky confidence below `sky_threshold` are classified as sky
- Sky pixels are filled with a depth value derived from the 99th percentile of valid (non-sky) depths
- The fill value is capped at `sky_depth_cap` meters to avoid unrealistic far distances

### 4. Resolution Upscaling
- The depth map is resized from model resolution back to the original camera resolution using cubic interpolation

### 5. Point Cloud Generation
For each pixel in the depth map:
```
X = (u - cx) * depth / fx
Y = (v - cy) * depth / fy
Z = depth
```
Where `(cx, cy)` is the principal point and `(fx, fy)` are the focal lengths from `CameraInfo`.

Optional features:
- **Downsampling**: Use `point_cloud_downsample_factor` to reduce point count for visualization or bandwidth
- **Colorization**: When `colorize_point_cloud` is enabled, RGB values from the input image are added to each 3D point

When `publish_compressed_depth` is enabled, the node advertises the base depth topic `depth_image/raw` with `image_transport` and internally restricts publisher plugins to `image_transport/raw` and `image_transport/compressedDepth`. This avoids `image_transport/compressed` JPEG handling, which is not appropriate for 32-bit depth images and can prevent publication.


## Troubleshooting

### Common Issues

1. **TensorRT engine building fails**:
   - Check CUDA/TensorRT compatibility
   - Verify ONNX model format
   - Increase workspace size

2. **Point cloud appears incorrect**:
   - Verify camera_info calibration
   - Check coordinate frame conventions
   - Validate depth value units

3. **Performance issues**:
   - Enable FP16 precision
   - Check GPU memory usage

### Debug Mode

Enable debug mode to troubleshoot:
```yaml
enable_debug: true
debug_colormap: "JET"
debug_colormap_min_depth: 0.0
debug_colormap_max_depth: 50.0
write_colormap: true
```

This will publish colorized depth images and save them to disk for inspection.


## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite the following:

```bibtex
@misc{beemelmanns2024depth,
  author = {Till Beemelmanns},
  title = {ros2-depth-anything-v3-trt: ROS2 TensorRT Node for Monocular Metric Depth estimation and Point Cloud generation with Depth Anything V3},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt}
}
```

## Acknowledgements
Thanks to the following repositories for inspiration:

- [ROS 2 MoGE TRT Node](https://github.com/ika-rwth-aachen/ros2-moge-trt/tree/main)
- [Depth-Anything-V3](https://github.com/ByteDance-Seed/depth-anything-3)
- [Monocular_Depth_Estimation_TRT](https://github.com/yester31/Monocular_Depth_Estimation_TRT)
- [DepthAnything-ROS](https://github.com/scepter914/DepthAnything-ROS)
