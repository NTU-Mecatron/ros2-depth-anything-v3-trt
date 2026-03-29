# Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pkg_dir = get_package_share_directory('depth_anything_v3')
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'depth_anything_v3.param.yaml'),
        description='Path to the parameter file'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='depth_anything_v3',
        description='Namespace for the Depth Anything V3 component node.'
    )

    container = ComposableNodeContainer(
        name='depth_anything_v3_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_anything_v3',
                plugin='depth_anything_v3::DepthAnythingV3Node',
                name='depth_anything_v3',
                namespace=namespace,
                parameters=[params_file],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        params_file_arg,
        namespace_arg,
        container,
    ])
