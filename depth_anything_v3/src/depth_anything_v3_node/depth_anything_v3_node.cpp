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

#include "depth_anything_v3/depth_anything_v3_node.hpp"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });

  // Not found
  if (itr == params.cend()) {
    return false;
  }

  value = itr->template get_value<T>();
  return true;
}
} // namespace

namespace depth_anything_v3
{
using namespace std::literals;

DepthAnythingV3Node::DepthAnythingV3Node(const rclcpp::NodeOptions & node_options)
: Node("depth_anything_v3", node_options)
{
  using std::placeholders::_1;
  
  // Parameter
  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&DepthAnythingV3Node::onSetParam, this, _1));
  
  node_param_.onnx_path = declare_parameter<std::string>(
    "onnx_path", "models/DA3METRIC-LARGE.fp16-batch1.engine");
  node_param_.precision = declare_parameter<std::string>("precision", "fp16");
  node_param_.sky_threshold = declare_parameter<double>("sky_threshold", 0.3);
  node_param_.sky_depth_cap = declare_parameter<double>("sky_depth_cap", 200.0);
  node_param_.input_image_topic = declare_parameter<std::string>("input_image_topic", "");
  node_param_.input_camera_info_topic =declare_parameter<std::string>("input_camera_info_topic", "");
  node_param_.output_depth_topic = declare_parameter<std::string>("output_depth_topic", "");
  node_param_.output_point_cloud_topic = declare_parameter<std::string>("output_point_cloud_topic", "");
  if (node_param_.input_image_topic.empty() || node_param_.input_camera_info_topic.empty()) {
    throw std::runtime_error(
      "Input topic parameters 'input_image_topic' and 'input_camera_info_topic' must be non-empty.");
  }
  if (node_param_.output_depth_topic.empty()) {
    node_param_.output_depth_topic = "output/depth_image";
  }
  if (node_param_.output_point_cloud_topic.empty()) {
    node_param_.output_point_cloud_topic = "output/point_cloud";
  }
  
  // Point cloud parameters
  node_param_.publish_point_cloud = declare_parameter<bool>("publish_point_cloud", true);
  node_param_.point_cloud_downsample_factor = declare_parameter<int>("point_cloud_downsample_factor", 10);
  node_param_.colorize_point_cloud = declare_parameter<bool>("colorize_point_cloud", true);
  if (node_param_.point_cloud_downsample_factor < 1) {
    throw std::runtime_error(
      "Parameter 'point_cloud_downsample_factor' must be >= 1.");
  }
  RCLCPP_INFO(
    get_logger(), "Point cloud publishing: %s",
    node_param_.publish_point_cloud ? "enabled" : "disabled");
  RCLCPP_INFO(get_logger(), "Point cloud downsampling factor: %d (publishing every %dth point)", 
    node_param_.point_cloud_downsample_factor, node_param_.point_cloud_downsample_factor);

  RCLCPP_INFO(get_logger(), "Using model file: %s", node_param_.onnx_path.c_str());

  // Create subscriptions
  sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
    node_param_.input_image_topic, rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onImage, this, _1));
  sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    node_param_.input_camera_info_topic, rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onCameraInfo, this, _1));

  RCLCPP_INFO(get_logger(), "Depth Anything V3 TensorRT node initialized successfully");
  RCLCPP_INFO(get_logger(), "Waiting for input messages on:");
  RCLCPP_INFO(get_logger(), "  - Image topic: %s", node_param_.input_image_topic.c_str());
  RCLCPP_INFO(get_logger(), "  - Camera info topic: %s", node_param_.input_camera_info_topic.c_str());

  // Publishers
  pub_depth_image_ = create_publisher<sensor_msgs::msg::Image>(node_param_.output_depth_topic, 1);
  pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(node_param_.output_point_cloud_topic, 1);
  RCLCPP_INFO(get_logger(), "Publishing depth image on: %s", node_param_.output_depth_topic.c_str());
  RCLCPP_INFO(get_logger(), "Publishing point cloud on: %s", node_param_.output_point_cloud_topic.c_str());

  // Init TensorRT model
  std::string calibType = "MinMax";
  int dla = -1;
  bool first = false;
  bool last = false;
  bool prof = false;
  double clip = 0.0;
  tensorrt_common::BuildConfig build_config(calibType, dla, first, last, prof, clip);

  int batch = 1;
  tensorrt_common::BatchConfig batch_config{1, batch / 2, batch};

  bool use_gpu_preprocess = false;
  std::string calibration_images = "calibration_images.txt";
  const size_t workspace_size = (1 << 30);

  tensorrt_depth_anything_ = std::make_shared<TensorRTDepthAnything>(
    node_param_.onnx_path, node_param_.precision, build_config, use_gpu_preprocess,
    calibration_images, batch_config, workspace_size);
  tensorrt_depth_anything_->setSkyThreshold(static_cast<float>(node_param_.sky_threshold));
  tensorrt_depth_anything_->setSkyDepthCap(static_cast<float>(node_param_.sky_depth_cap));
    
  RCLCPP_INFO(get_logger(), "Finished initializing Depth Anything V3 TensorRT model");
}

void DepthAnythingV3Node::onImage(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
{
  const auto camera_info_msg = std::atomic_load(&latest_camera_info_);
  if (!camera_info_msg) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Skipping image because no camera_info has been received yet");
    return;
  }

  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  if (!is_initialized_) {
    RCLCPP_INFO(get_logger(), "Initializing TensorRT preprocessing buffer for %dx%d images", width, height);
    tensorrt_depth_anything_->initPreprocessBuffer(width, height);
    is_initialized_ = true;
    RCLCPP_INFO(get_logger(), "TensorRT preprocessing buffer initialized");
  }

  std::vector<cv::Mat> input_images;
  input_images.push_back(in_image_ptr->image);
  
  auto start = std::chrono::high_resolution_clock::now();
  bool success = tensorrt_depth_anything_->doInference(
    input_images, *camera_info_msg, node_param_.publish_point_cloud,
    node_param_.point_cloud_downsample_factor, node_param_.colorize_point_cloud);
  auto end = std::chrono::high_resolution_clock::now();
  const double inference_time_sec = std::chrono::duration<double>(end - start).count();
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Depth Anything V3 inference FAILED!");
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Inference completed in %.3f ms", inference_time_sec * 1000.0);

  // Get depth image result
  const cv::Mat& depth_image = tensorrt_depth_anything_->getDepthImage();
  
  // Publish depth image (32FC1 format for accurate depth values)
  cv_bridge::CvImage cv_img_depth;
  cv_img_depth.image = depth_image;
  cv_img_depth.encoding = "32FC1";
  cv_img_depth.header = image_msg->header;
  pub_depth_image_->publish(*cv_img_depth.toImageMsg());

  if (node_param_.publish_point_cloud) {
    sensor_msgs::msg::PointCloud2 point_cloud = tensorrt_depth_anything_->getPointCloud();
    point_cloud.header = image_msg->header;
    pub_point_cloud_->publish(point_cloud);
  }
}

void DepthAnythingV3Node::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{
  std::atomic_store(&latest_camera_info_, std::shared_ptr<const sensor_msgs::msg::CameraInfo>(camera_info_msg));
}

rcl_interfaces::msg::SetParametersResult DepthAnythingV3Node::onSetParam(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  try {
    auto & p = node_param_;
    int new_point_cloud_downsample_factor = p.point_cloud_downsample_factor;

    for (const auto & param : params) {
      const auto & name = param.get_name();
      if (
        (name == "input_image_topic" || name == "input_camera_info_topic" ||
        name == "output_depth_topic" || name == "output_point_cloud_topic") &&
        (sub_image_ || sub_camera_info_ || pub_depth_image_ || pub_point_cloud_))
      {
        result.successful = false;
        result.reason = "Topic parameters are startup-only and cannot be changed at runtime.";
        return result;
      }

      if (name == "point_cloud_downsample_factor") {
        new_point_cloud_downsample_factor = param.as_int();
      }
    }

    if (new_point_cloud_downsample_factor < 1) {
      result.successful = false;
      result.reason = "Parameter 'point_cloud_downsample_factor' must be >= 1.";
      return result;
    }
    
    // Update all parameters uniformly
    update_param(params, "onnx_path", p.onnx_path);
    update_param(params, "precision", p.precision);
    update_param(params, "sky_threshold", p.sky_threshold);
    update_param(params, "sky_depth_cap", p.sky_depth_cap);
    update_param(params, "publish_point_cloud", p.publish_point_cloud);
    update_param(params, "point_cloud_downsample_factor", p.point_cloud_downsample_factor);
    update_param(params, "colorize_point_cloud", p.colorize_point_cloud);
    
    // Apply runtime-configurable model parameters
    if (tensorrt_depth_anything_) {
      tensorrt_depth_anything_->setSkyThreshold(static_cast<float>(p.sky_threshold));
      tensorrt_depth_anything_->setSkyDepthCap(static_cast<float>(p.sky_depth_cap));
    }
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
    return result;
  }
  result.successful = true;
  result.reason = "success";
  return result;
}

} // namespace depth_anything_v3

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(depth_anything_v3::DepthAnythingV3Node)
