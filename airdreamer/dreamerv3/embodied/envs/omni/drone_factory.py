"""
drone_factory.py — 公共无人机工厂函数。

只暴露 make_drone(drone_model, controller, device, name) 一个函数，返回 (drone, ctrl)。
不引入 arena/env/sensor 依赖，不处理 scene 和 video。
"""


def make_drone(drone_model, controller, device, name):
    """解析无人机型号并实例化控制器。

    Args:
        drone_model: 无人机型号名称字符串。
        controller:  控制器名称字符串，None 表示不使用控制器。
        device:      torch 设备字符串，例如 'cuda:0'。
        name:        无人机实例名称字符串。

    Returns:
        (drone, ctrl) 元组。ctrl 在 controller=None 时为 None。
    """
    from omni_drones.controllers import ControllerBase
    from omni_drones.robots.drone.multirotor import MultirotorBase

    drone_cls = MultirotorBase.REGISTRY[str(drone_model)]
    drone = drone_cls(name=str(name))
    ctrl = None
    if controller is not None:
        controller_cls = ControllerBase.REGISTRY[str(controller)]
        ctrl = controller_cls(drone.gravity[1], drone.params).to(device)
    return drone, ctrl
