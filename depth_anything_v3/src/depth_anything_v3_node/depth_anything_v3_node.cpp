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
#include <algorithm>
#include <vector>
#include <chrono>
#include <functional>
#include <stdexcept>

namespace
{
constexpr char kInputImageTopic[] = "image/raw";
constexpr char kInputCameraInfoTopic[] = "camera_info";
constexpr char kOutputDepthTopic[] = "depth_image/raw";
constexpr char kOutputPointCloudTopic[] = "point_cloud";
const std::vector<std::string> kDepthImageTransportPubPlugins{
  "image_transport/raw", "image_transport/compressedDepth"};

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
DepthAnythingV3Node::DepthAnythingV3Node(const rclcpp::NodeOptions & node_options)
: rclcpp_lifecycle::LifecycleNode("depth_anything_v3", node_options)
{
  using std::placeholders::_1;

  node_param_.onnx_path = declare_parameter<std::string>(
    "onnx_path", "models/DA3METRIC-LARGE.fp16-batch1.engine");
  node_param_.precision = declare_parameter<std::string>("precision", "fp16");
  node_param_.sky_threshold = declare_parameter<double>("sky_threshold", 0.3);
  node_param_.sky_depth_cap = declare_parameter<double>("sky_depth_cap", 200.0);
  node_param_.publish_compressed_depth = declare_parameter<bool>("publish_compressed_depth", false);
  node_param_.publish_point_cloud = declare_parameter<bool>("publish_point_cloud", true);
  node_param_.point_cloud_downsample_factor = declare_parameter<int>("point_cloud_downsample_factor", 10);
  node_param_.colorize_point_cloud = declare_parameter<bool>("colorize_point_cloud", true);

  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&DepthAnythingV3Node::onSetParam, this, _1));

  RCLCPP_INFO(
    get_logger(),
    "Depth Anything V3 lifecycle node constructed; awaiting configure transition");
}

CallbackReturn DepthAnythingV3Node::on_configure(const rclcpp_lifecycle::State &)
{
  get_parameter("onnx_path", node_param_.onnx_path);
  get_parameter("precision", node_param_.precision);
  get_parameter("sky_threshold", node_param_.sky_threshold);
  get_parameter("sky_depth_cap", node_param_.sky_depth_cap);
  get_parameter("publish_compressed_depth", node_param_.publish_compressed_depth);
  get_parameter("publish_point_cloud", node_param_.publish_point_cloud);
  get_parameter("point_cloud_downsample_factor", node_param_.point_cloud_downsample_factor);
  get_parameter("colorize_point_cloud", node_param_.colorize_point_cloud);

  if (node_param_.point_cloud_downsample_factor < 1) {
    RCLCPP_ERROR(get_logger(), "Parameter 'point_cloud_downsample_factor' must be >= 1.");
    return CallbackReturn::FAILURE;
  }

  sub_image_.reset();
  sub_camera_info_.reset();
  pub_depth_image_transport_.shutdown();
  image_transport_node_.reset();
  pub_depth_image_.reset();
  pub_point_cloud_.reset();
  tensorrt_depth_anything_.reset();
  std::atomic_store(&latest_camera_info_, std::shared_ptr<const sensor_msgs::msg::CameraInfo>{});
  is_initialized_ = false;

  RCLCPP_INFO(get_logger(), "Using model file: %s", node_param_.onnx_path.c_str());
  RCLCPP_INFO(
    get_logger(), "Compressed depth publishing: %s",
    node_param_.publish_compressed_depth ? "enabled" : "disabled");
  RCLCPP_INFO(
    get_logger(), "Point cloud publishing: %s",
    node_param_.publish_point_cloud ? "enabled" : "disabled");
  RCLCPP_INFO(
    get_logger(), "Point cloud downsampling factor: %d (publishing every %dth point)",
    node_param_.point_cloud_downsample_factor, node_param_.point_cloud_downsample_factor);

  try {
    std::string calib_type = "MinMax";
    int dla = -1;
    bool first = false;
    bool last = false;
    bool prof = false;
    double clip = 0.0;
    tensorrt_common::BuildConfig build_config(calib_type, dla, first, last, prof, clip);

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
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to configure TensorRT model: %s", e.what());
    tensorrt_depth_anything_.reset();
    return CallbackReturn::FAILURE;
  }

  if (node_param_.publish_compressed_depth) {
    auto image_transport_node_options = get_node_options();
    image_transport_node_options.arguments({});
    image_transport_node_options.context(get_node_options().context());
    image_transport_node_options.use_intra_process_comms(
      get_node_options().use_intra_process_comms());
    image_transport_node_options.start_parameter_services(false);
    image_transport_node_options.start_parameter_event_publisher(false);
    image_transport_node_options.enable_rosout(false);

    image_transport_node_ = std::make_shared<rclcpp::Node>(
      std::string(get_name()) + "_image_transport_helper",
      get_namespace(),
      image_transport_node_options);
    image_transport_node_->declare_parameter<std::vector<std::string>>(
      "depth_image.raw.enable_pub_plugins", kDepthImageTransportPubPlugins);

    auto qos = rmw_qos_profile_default;
    qos.depth = 1;
    pub_depth_image_transport_ = image_transport::create_publisher(
      image_transport_node_.get(), kOutputDepthTopic, qos);
  } else {
    pub_depth_image_ = create_publisher<sensor_msgs::msg::Image>(
      kOutputDepthTopic, rclcpp::QoS(1));
  }
  pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    kOutputPointCloudTopic, rclcpp::QoS(1));

  RCLCPP_INFO(get_logger(), "Configured Depth Anything V3 lifecycle node");
  RCLCPP_INFO(get_logger(), "  - Image topic: %s", kInputImageTopic);
  RCLCPP_INFO(get_logger(), "  - Camera info topic: %s", kInputCameraInfoTopic);
  RCLCPP_INFO(get_logger(), "  - Depth output topic: %s", kOutputDepthTopic);
  RCLCPP_INFO(
    get_logger(), "  - Point cloud output topic: %s",
    kOutputPointCloudTopic);
  RCLCPP_INFO(
    get_logger(),
    "  - Use standard ROS remapping rules to change topic names");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DepthAnythingV3Node::on_activate(const rclcpp_lifecycle::State &)
{
  using std::placeholders::_1;

  if (pub_depth_image_) {
    pub_depth_image_->on_activate();
  }
  if (pub_point_cloud_) {
    pub_point_cloud_->on_activate();
  }

  sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
    kInputImageTopic, rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onImage, this, _1));
  sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    kInputCameraInfoTopic, rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onCameraInfo, this, _1));

  RCLCPP_INFO(get_logger(), "Activated Depth Anything V3 lifecycle node");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DepthAnythingV3Node::on_deactivate(const rclcpp_lifecycle::State &)
{
  sub_image_.reset();
  sub_camera_info_.reset();
  std::atomic_store(&latest_camera_info_, std::shared_ptr<const sensor_msgs::msg::CameraInfo>{});
  is_initialized_ = false;

  if (pub_depth_image_) {
    pub_depth_image_->on_deactivate();
  }
  if (pub_point_cloud_) {
    pub_point_cloud_->on_deactivate();
  }

  RCLCPP_INFO(get_logger(), "Deactivated Depth Anything V3 lifecycle node");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DepthAnythingV3Node::on_cleanup(const rclcpp_lifecycle::State &)
{
  sub_image_.reset();
  sub_camera_info_.reset();
  pub_depth_image_transport_.shutdown();
  image_transport_node_.reset();
  pub_depth_image_.reset();
  pub_point_cloud_.reset();
  tensorrt_depth_anything_.reset();
  std::atomic_store(&latest_camera_info_, std::shared_ptr<const sensor_msgs::msg::CameraInfo>{});
  is_initialized_ = false;

  RCLCPP_INFO(get_logger(), "Cleaned up Depth Anything V3 lifecycle node");
  return CallbackReturn::SUCCESS;
}

CallbackReturn DepthAnythingV3Node::on_shutdown(const rclcpp_lifecycle::State &)
{
  sub_image_.reset();
  sub_camera_info_.reset();
  pub_depth_image_transport_.shutdown();
  image_transport_node_.reset();
  pub_depth_image_.reset();
  pub_point_cloud_.reset();
  tensorrt_depth_anything_.reset();
  std::atomic_store(&latest_camera_info_, std::shared_ptr<const sensor_msgs::msg::CameraInfo>{});
  is_initialized_ = false;

  RCLCPP_INFO(get_logger(), "Shut down Depth Anything V3 lifecycle node");
  return CallbackReturn::SUCCESS;
}

void DepthAnythingV3Node::onImage(const sensor_msgs::msg::Image::ConstSharedPtr & image_msg)
{
  const bool has_depth_publisher = node_param_.publish_compressed_depth || static_cast<bool>(pub_depth_image_);
  const bool has_point_cloud_publisher = !node_param_.publish_point_cloud || static_cast<bool>(pub_point_cloud_);
  if (!tensorrt_depth_anything_ || !has_depth_publisher || !has_point_cloud_publisher) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Skipping image because lifecycle resources are not configured");
    return;
  }

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
  auto depth_msg = std::make_unique<sensor_msgs::msg::Image>();
  cv_img_depth.toImageMsg(*depth_msg);
  if (node_param_.publish_compressed_depth) {
    pub_depth_image_transport_.publish(std::move(depth_msg));
  } else {
    pub_depth_image_->publish(std::move(depth_msg));
  }

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
        (
          name == "onnx_path" || name == "precision" ||
          name == "publish_compressed_depth") &&
        (tensorrt_depth_anything_ || image_transport_node_ || pub_depth_image_ || pub_point_cloud_))
      {
        result.successful = false;
        result.reason =
          "onnx_path, precision, and publish_compressed_depth are configure-time-only parameters.";
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
    update_param(params, "publish_compressed_depth", p.publish_compressed_depth);
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
