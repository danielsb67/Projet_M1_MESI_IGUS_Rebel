import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PointStamped, PoseStamped
from control_msgs.action import GripperCommand

import tf2_ros
from tf2_geometry_msgs import do_transform_point

# MoveIt2 python interface (common in ROS2):
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface


class PickWithMoveIt(Node):
    def __init__(self):
        super().__init__("pick_with_moveit")

        # -------------------------
        # 1) Topics / frames
        # -------------------------
        self.object_topic = "/object_position_in_world"
        self.base_frame = "base_link"

        # -------------------------
        # 2) MoveIt setup (EDIT THESE)
        # -------------------------
        self.ARM_GROUP = "arm"          # <-- change to your group name (e.g. "manipulator")
        self.EE_LINK = ""              # <-- optional: set to your EE link (e.g. "tool0")

        # MoveIt objects
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander(self.ARM_GROUP)

        if self.EE_LINK:
            self.group.set_end_effector_link(self.EE_LINK)

        # basic planning settings (optional)
        self.group.set_planning_time(3.0)
        self.group.set_num_planning_attempts(5)

        # -------------------------
        # 3) TF2
        # -------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------------------------
        # 4) Gripper action client
        # -------------------------
        self.gripper = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd")
        self.open_gap = 0.006     # from your URDF: max gap ~ 0.006 m
        self.close_gap = 0.0
        self.max_effort = 40.0

        # -------------------------
        # 5) Pick parameters
        # -------------------------
        self.pregrasp_z = 0.08
        self.grasp_z_offset = 0.00
        self.lift_z = 0.10

        # -------------------------
        # 6) State machine
        # -------------------------
        self.state = "WAIT_OBJECT"
        self.object_base = None  # PointStamped in base frame

        self.sub = self.create_subscription(PointStamped, self.object_topic, self.object_cb, 10)
        self.timer = self.create_timer(0.1, self.step)  # 10 Hz

        self.get_logger().info("PickWithMoveIt started.")

    # -------------------------
    # Callbacks
    # -------------------------
    def object_cb(self, msg: PointStamped):
        # transform object to base_link
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, rclpy.time.Time())
            self.object_base = do_transform_point(msg, tf)
        except Exception:
            return

    # -------------------------
    # Helpers
    # -------------------------
    def send_gripper(self, gap):
        # GripperCommand expects "position" (often q), not full gap
        q = gap / 2.0  # because gap = 2*q in your URDF

        goal = GripperCommand.Goal()
        goal.command.position = float(q)
        goal.command.max_effort = float(self.max_effort)

        if not self.gripper.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Gripper action server not available")
            return

        self.gripper.send_goal_async(goal)

    def moveit_go_to_pose(self, pose: PoseStamped) -> bool:
        """Plan + execute to a pose target. Returns True if success."""
        self.group.set_pose_target(pose)

        ok = self.group.go(wait=True)  # plan + execute
        self.group.stop()
        self.group.clear_pose_targets()
        return bool(ok)

    def make_pose(self, x, y, z) -> PoseStamped:
        """Simple pose with fixed orientation (EDIT orientation for your gripper)."""
        p = PoseStamped()
        p.header.frame_id = self.base_frame
        p.header.stamp = self.get_clock().now().to_msg()

        p.pose.position.x = float(x)
        p.pose.position.y = float(y)
        p.pose.position.z = float(z)

        # IMPORTANT: set an orientation that matches your gripper
        # This is identity quaternion (no rotation). You will likely need to change it.
        p.pose.orientation.w = 1.0

        return p

    # -------------------------
    # State machine
    # -------------------------
    def step(self):
        if self.state == "WAIT_OBJECT":
            if self.object_base is None:
                return
            self.get_logger().info("Object received -> opening gripper")
            self.send_gripper(self.open_gap)
            self.state = "MOVE_PREGRASP"

        elif self.state == "MOVE_PREGRASP":
            p = self.object_base.point
            pre = self.make_pose(p.x, p.y, p.z + self.pregrasp_z)

            self.get_logger().info("MoveIt -> pregrasp")
            if self.moveit_go_to_pose(pre):
                self.state = "MOVE_GRASP"
            else:
                self.get_logger().warn("Pregrasp failed -> wait object")
                self.state = "WAIT_OBJECT"

        elif self.state == "MOVE_GRASP":
            p = self.object_base.point
            grasp = self.make_pose(p.x, p.y, p.z + self.grasp_z_offset)

            self.get_logger().info("MoveIt -> grasp")
            if self.moveit_go_to_pose(grasp):
                self.get_logger().info("Closing gripper")
                self.send_gripper(self.close_gap)
                self.state = "LIFT"
            else:
                self.get_logger().warn("Grasp approach failed -> wait object")
                self.state = "WAIT_OBJECT"

        elif self.state == "LIFT":
            p = self.object_base.point
            lift = self.make_pose(p.x, p.y, p.z + self.lift_z)

            self.get_logger().info("MoveIt -> lift")
            if self.moveit_go_to_pose(lift):
                self.state = "HOME"
            else:
                self.get_logger().warn("Lift failed -> HOME anyway")
                self.state = "HOME"

        elif self.state == "HOME":
            self.get_logger().info("MoveIt -> home (named target)")
            # If you have a named target in SRDF (common): "home"
            try:
                self.group.set_named_target("home")
                ok = self.group.go(wait=True)
                self.group.stop()
                self.state = "WAIT_OBJECT"
            except Exception:
                self.get_logger().warn("No named target 'home' configured. Staying in WAIT_OBJECT.")
                self.state = "WAIT_OBJECT"


def main():
    rclpy.init()
    node = PickWithMoveIt()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
