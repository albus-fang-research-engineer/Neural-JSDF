import numpy as np
from fk_helper import RobotSDFHelper

urdf_path = "/root/Neural-JSDF/ur5e/robot_models/ur5e/ur5e.urdf"

robot = RobotSDFHelper(urdf_path)

config = np.array([-0.95377294, -0.81537088, -0.20763856,  0.17086699, -0.56765966,  0.81209045])
point = np.array([0.44492641, -0.38576019, 0.50026786])
config = np.array([-0.90954603, -0.84138828,  0.63966228, -0.20997181, -0.17360593, -0.41514037])
point  = np.array([ 0.1199757 , -0.2554397 ,  0.65242409])
robot.set_q(config)
robot.visualize(point)