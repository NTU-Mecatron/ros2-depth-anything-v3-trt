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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution([FindPackageShare('depth_anything_v3'), 'config', 'depth_anything_v3.param.yaml'])

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
                namespace=LaunchConfiguration('namespace'),
                parameters=[config_file],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        namespace_arg,
        container,
    ])
