// Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_
#define DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "depth_anything_v3/tensorrt_depth_anything.hpp"

namespace depth_anything_v3
{
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class DepthAnythingV3Node : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit DepthAnythingV3Node(const rclcpp::NodeOptions & node_options);

  struct NodeParam
  {
    std::string onnx_path{};
    std::string precision{};
    double sky_threshold{};             // Threshold for sky classification
    double sky_depth_cap{};             // Cap for sky depth fill-in
    bool publish_point_cloud{};         // Whether to generate and publish point clouds
    int point_cloud_downsample_factor{};  // Only publish every Nth point (1 = no downsampling)
    bool colorize_point_cloud{};  // Add RGB colors from input image to point cloud
  };

private:
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_camera_info_;

  // Callbacks
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg);
  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);

  // Publishers
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud_;

  // Parameter Server
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult onSetParam(
    const std::vector<rclcpp::Parameter> & params);

  // Parameter
  NodeParam node_param_{};

  // Core
  std::shared_ptr<TensorRTDepthAnything> tensorrt_depth_anything_;
  std::shared_ptr<const sensor_msgs::msg::CameraInfo> latest_camera_info_;
  bool is_initialized_ = false;
};

} // namespace depth_anything_v3

#endif // DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_
