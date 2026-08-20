#!/usr/bin/env python3

from enum import Enum

import rclpy
from rclpy.node import Node, Optional
from rclpy.executors import Future, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from yolo_msgs.srv import SetClasses
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from yolo_msgs.msg import DetectionArray, Detection
from geometry_msgs.msg import Point, QuaternionStamped
from tf_transformations import quaternion_from_euler
import math

from smarc_action_base.gentler_action_server import GentlerActionServer


class YoloActionServer:
    def __init__(self, node: Node):
        self._node = node

        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()

        self._node.declare_parameter('set_classes_serivce', "/yolo/set_classes")
        set_classes_serivce_name = self._node.get_parameter('set_classes_serivce').value

        self._node.declare_parameter('set_parameter_serivce', "/yolo/yolo_node/set_parameters")
        set_parameter_serivce_name = self._node.get_parameter('set_parameter_serivce').value

        self._node.declare_parameter('yolo_tracking_topic', "/yolo/tracking")
        yolo_tracking_topic = self._node.get_parameter('yolo_tracking_topic').value

        self._node.declare_parameter('image_poi_output', "/yolo/tracked_poi")
        image_poi_output = self._node.get_parameter('image_poi_output').value

        # Single-target output. yolo_action is the place that decides which
        # tracked object to follow; it republishes that choice here so every
        # downstream consumer (gimbal + vehicle backend) agrees on the same id
        self._node.declare_parameter('target_output', "/yolo/target")
        target_output = self._node.get_parameter('target_output').value

        # How long (seconds) a locked id may be absent before we release it and
        # re-acquire.
        self._node.declare_parameter('lock_timeout', 3.0)
        self._lock_timeout = self._node.get_parameter('lock_timeout').get_parameter_value().double_value

        self._node.declare_parameter('camera_aperture', 57.1)
        self.camera_aperture = self._node.get_parameter('camera_aperture').get_parameter_value().double_value

        self._node.declare_parameter('camera_frame_id', "evolo/z1_camera_link")
        self.camera_frame_id = self._node.get_parameter('camera_frame_id').value

        # Classes to detect at startup.
        self._node.declare_parameter('startup_classes', [""])
        self._startup_classes = [c for c in self._node.get_parameter('startup_classes').value if c]

        # --- Target-selection state ---
        #   MANUAL : _manual_id is set -> follow EXACTLY that track id. If it is
        #            lost we wait for it indefinitely and NEVER fall back to
        #            another object. Only an explicit AUTO / -1 request leaves
        #            this mode.
        #   AUTO   : _manual_id is None (default) -> acquire the highest-confidence
        #            track, lock onto it, and on loss > lock_timeout re-acquire
        #            the next highest.
        # _locked_id : the auto-mode lock (str), or None to (re)acquire highest.
        self._manual_id: str | None = None
        self._locked_id: str | None = None
        self._last_seen: float = 0.0

        #subscribers
        self.tracking_subscriber = node.create_subscription(DetectionArray, yolo_tracking_topic, self.yolo_tracking_cb, 10)

        #publishers
        self.image_poi_publisher = node.create_publisher(QuaternionStamped, image_poi_output, 10)
        # Chosen-target stream consumed by the vehicle backend (0 or 1 detection).
        self.target_publisher = node.create_publisher(DetectionArray, target_output, 10)

        #Service clients
        self.set_classes_client = self._node.create_client(SetClasses, set_classes_serivce_name, callback_group=self.service_callback_group)
        self.set_threshold_client = self._node.create_client(SetParameters,set_parameter_serivce_name, callback_group=self.service_callback_group)

        # Wait for service
        while not self.set_classes_client.wait_for_service(timeout_sec=1.0): #Does this waiting cause problems?
            self._node.get_logger().info('set classes service not available, waiting...')

        # Wait for service
        while not self.set_threshold_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info('set param service not available, waiting...')
        
        # The timer callback function will make the service call if the request if not None
        # (for thread reasons) 
        self.set_classes_request = None
        self.set_threshold_request = None
        
        #Futures for keeping track of service calls
        self.set_classes_future = None
        self.set_threshold_future = None

        # Push startup classes
        if self._startup_classes:
            self.set_classes_request = SetClasses.Request()
            self.set_classes_request.classes = self._startup_classes
            self._node.get_logger().info(f"Queued startup classes: {self._startup_classes}")


        self._classes_as = GentlerActionServer(
            self._node,
            "yolo_set_classes",
            self._on_goal_received_classes,
            lambda: True,
            lambda: None,
            lambda: True,
            lambda: "No feedback",
            loop_frequency = 1.0
        )

        # NOTE: previously this was also assigned to self._classes_as, which
        # clobbered the classes action-server handle. It is its own server.
        self._threshold_as = GentlerActionServer(
            self._node,
            "yolo_set_threshold",
            self._on_goal_received_threshold,
            lambda: True,
            lambda: None,
            lambda: True,
            lambda: "No feedback",
            loop_frequency = 1.0
        )

        self._tracking_as = GentlerActionServer(
            self._node,
            "yolo_set_tracking",
            self._on_goal_received_tracking,
            lambda: True,
            lambda: None,
            lambda: True,
            lambda: "No feedback",
            loop_frequency = 1.0
        )

        timer = node.create_timer(1.0 , self.timer_cb, callback_group=self.timer_callback_group)

        self._node.get_logger().info(f"YoloServer initialized.")

    #Callback server for printing the result of a service call
    def service_callback_response(self, future):
        try:
            response = future.result()
            self._node.get_logger().info(f'Result: {response}')
        except Exception as e:
            self._node.get_logger().error(f'Service call failed: {e}')

    def timer_cb(self):
        self._node.get_logger().warn(f"Timer callback")
        
        #Set classes 
        if(self.set_classes_request is not None):
            self._node.get_logger().info(f"Calling set classes service")
            #Check if we are currently trying to do a service call. Anc cancel it in that case
            if not (self.set_classes_future is None or self.set_classes_future.done):
                self._node.get_logger().error(f'Service call was not finnished before next call. Canceling service call')
                self.set_classes_future.cancel()

            # Make async call
            self.set_classes_future = self.set_classes_client.call_async(self.set_classes_request)

            # Attach callback
            self.set_classes_future.add_done_callback(self.service_callback_response)
            
            # Clear request so we donn't call the service next time
            self.set_classes_request = None

        #Set threshold 
        if(self.set_threshold_request is not None):
            self._node.get_logger().info(f"Calling threshold param service")

            #Check if we are currently trying to do a service call. Anc cancel it in that case
            if not (self.set_classes_future is None or self.set_classes_future.done):
                self._node.get_logger().error(f'Service call was not finnished before next call. Canceling service call')
                self.set_classes_future.cancel()
            
            # Make async call
            self.set_threshold_future = self.set_threshold_client.call_async(self.set_threshold_request)

            # Attach callback
            self.set_threshold_future.add_done_callback(self.service_callback_response)
            
            # Clear request so we donn't call the service next time
            self.set_threshold_request = None
           
    def yolo_tracking_cb(self, msg : DetectionArray):
        """
        Runs on every /yolo/tracking message

        Two things happen here:
          1. _select_target() applies the lock / timeout / re-acquire policy and
             returns the ONE detection we are currently committed to (or None).
          2. We publish that choice on two topics:
               - /yolo/target       (DetectionArray, 0 or 1 entries) so the
                 vehicle backend follows this object.
               - /yolo/tracked_poi  (QuaternionStamped) the camera-frame bearing
                 the gimbal action server turns toward.
        While target is None we publish an EMPTY /yolo/target and no poi, letting
        downstream nodes coast on their own staleness logic through brief losses.
        """
        try:
            self._yolo_tracking_cb_inner(msg)
        except Exception as e:
            self._node.get_logger().error(f"Exception in yolo tracking callback: {e}")

    def _yolo_tracking_cb_inner(self, msg: DetectionArray):
        target = self._select_target(msg.detections)

        # (1) Republish the chosen target (possibly empty) for the vehicle backend.
        target_msg = DetectionArray()
        target_msg.header = msg.header
        if target is not None:
            target_msg.detections.append(target)
        self.target_publisher.publish(target_msg)

        # Nothing committed this frame (no candidates, or locked id still inside
        # its grace window): don't move the gimbal, let it hold its last angle.
        if target is None:
            return

        # (2) Project the chosen bbox centre into a camera-frame bearing for the gimbal action server to follow.
        IMAGE_SIZE = (target.mask.width, target.mask.height)

        #math
        pixel_error_x = target.bbox.center.position.x - 0.5*IMAGE_SIZE[0]
        pixel_error_y = target.bbox.center.position.y - 0.5*IMAGE_SIZE[1]
        angle_per_pixel = math.radians(self.camera_aperture) / IMAGE_SIZE[0]

        roll = 0
        yaw_from_center = -1.0 * pixel_error_x * angle_per_pixel
        pitch_from_center = 1.0 * pixel_error_y * angle_per_pixel

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch_from_center, yaw_from_center)

        poi_msg = QuaternionStamped()
        poi_msg.header.stamp = self._node.get_clock().now().to_msg()
        poi_msg.header.frame_id = self.camera_frame_id
        poi_msg.quaternion.x = qx
        poi_msg.quaternion.y = qy
        poi_msg.quaternion.z = qz
        poi_msg.quaternion.w = qw

        self.image_poi_publisher.publish(poi_msg)

    def _now_s(self) -> float:
        """Current node-clock time in seconds (float)."""
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _select_target(self, detections):
        """
        Decide which single detection to follow.

        Only detections with a non-empty id are considered: an empty id means
        the tracker has not committed to the box yet (possible one-shot false
        positive), so we never lock onto — or fall back to — those.
        """
        now = self._now_s()

        # Committed tracks only.
        candidates = [d for d in detections if d.id != ""]

        # --- MANUAL mode: follow exactly the requested id, indefinitely ---
        if self._manual_id is not None:
            for det in candidates:
                if det.id == self._manual_id:
                    self._last_seen = now
                    return det
            # Requested id not visible this frame — wait for it; never auto-switch.
            return None

        # --- AUTO mode: follow the currently locked id, if any ---
        if self._locked_id is not None:
            for det in candidates:
                if det.id == self._locked_id:
                    self._last_seen = now
                    return det

            # Locked id not present this frame.
            if (now - self._last_seen) < self._lock_timeout:
                # Still inside the grace window — hold the lock, emit nothing.
                return None

            # Grace window expired — give up this id and re-acquire below.
            self._node.get_logger().info(
                f"Lock on id '{self._locked_id}' lost for > {self._lock_timeout:.1f}s "
                f"— re-acquiring highest-confidence target."
            )
            self._locked_id = None

        # --- No lock: acquire the highest-confidence committed track ---
        if not candidates:
            return None

        best = max(candidates, key=lambda d: d.score)
        self._locked_id = best.id
        self._last_seen = now
        self._node.get_logger().info(
            f"Locked onto id '{best.id}' ({best.class_name}, conf={best.score:.2f})."
        )
        return best

    def _on_goal_received_classes(self, goal_request: dict) -> bool:
        """
        # classes = ['person', 'car', 'etc]
        """
        self._node.get_logger().info(f"Received new classes to track: {goal_request}")
        try:            
            self.set_classes_request = SetClasses.Request()
            self.set_classes_request.classes = goal_request['classes']
            return True
        except KeyError as e:
            self._node.get_logger().info("Missing key in goal request")
            return False
        except ValueError as e:
            self._node.get_logger().info("Invalid value in goal request")
            return False
        
    def _on_goal_received_threshold(self, goal_request: dict) -> bool:
        """
        # threshold = 0.5
        """
        self._node.get_logger().info(f"Received new threshold parameter: {goal_request}")
        try:
            param = Parameter( 'threshold', Parameter.Type.DOUBLE, float(goal_request['threshold']))
            self.set_threshold_request = SetParameters.Request()
            self.set_threshold_request.parameters = [param.to_parameter_msg()]
            return True
        except KeyError as e:
            self._node.get_logger().info("Missing key in goal request")
            return False
        except ValueError as e:
            self._node.get_logger().info("Invalid value in goal request")
            return False
        
    def _on_goal_received_tracking(self, goal_request: dict) -> bool:
        """
        Choose which tracked object the whole pipeline follows.

        goal_request:
          {"id": 5}        -> MANUAL: lock onto track id 5 and follow it
                              exclusively. If lost, wait for it indefinitely;
                              never fall back to another object.
          {"id": "AUTO"}   -> AUTO: highest-confidence track, with lock + timeout
                              + re-acquire (the default mode).
          {"id": -1}       -> same as AUTO (convenience for numeric senders).
        """
        self._node.get_logger().info(f"Received new tracking request: {goal_request}")
        try:
            val = goal_request["id"]

            # AUTO / negative -> leave manual mode; re-acquire highest conf.
            is_auto = (isinstance(val, str) and val.strip().upper() == "AUTO") or \
                      (isinstance(val, (int, float)) and int(val) < 0)
            if is_auto:
                self._manual_id = None
                self._locked_id = None
                self._node.get_logger().info("Tracking mode: AUTO (highest confidence).")
            else:
                # MANUAL lock. Detection ids are published as str(int(...)) by
                # tracking_node, so normalise to that form for comparison.
                self._manual_id = str(int(val))
                self._locked_id = None          # auto-lock state unused in manual mode
                self._last_seen = self._now_s()
                self._node.get_logger().info(
                    f"Tracking MANUALLY locked to id '{self._manual_id}' "
                    f"(will wait for it; no auto-fallback)."
                )
            return True
        except (KeyError, TypeError) as e:
            self._node.get_logger().info(f"Missing/invalid 'id' in tracking request: {e}")
            return False
        except ValueError as e:
            self._node.get_logger().info(f"Invalid value in tracking request: {e}")
            return False

def main():
    rclpy.init()
    
    node = Node("yolo_action_server_node")
    action_server = YoloActionServer(node)
    
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()