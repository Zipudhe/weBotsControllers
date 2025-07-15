import numpy as np
from utils.utils import calculate_targets, generate_position
from controller import Keyboard, Robot, Lidar, Camera, Supervisor
from utils.io import show_lidar_img, write_lidar_object_data, write_matrix_data, show_camera_img
import utils.pre_processing as pp

# robot = Robot() # para uso real
robot = Supervisor() # para treinamento
if robot is None:
    print("Robô não está em modo de treinamento!")
    exit()
    
print("Code updated")


class PioneerControllers:
    default_speed = 2.0
    time_step = 64

    def __init__(self):
        front_left = robot.getDevice("front left wheel")
        front_right = robot.getDevice("front right wheel")
        rear_left = robot.getDevice("back left wheel")
        rear_right = robot.getDevice("back right wheel")

        if None in [front_left, front_right, rear_left, rear_right]:
            print("Error: One or more motors not found!")
            robot.cleanup()
            exit(1)

        self.front_left = robot.getDevice("front left wheel")
        self.front_right = robot.getDevice("front right wheel")
        self.rear_left = robot.getDevice("back left wheel")
        self.rear_right = robot.getDevice("back right wheel")

        # Set all motors to velocity control mode and infinite rotation
        for motor in [front_left, front_right, rear_left, rear_right]:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        self.camera: Camera = robot.getDevice("camera")
        self.camera_width = self.camera.getWidth()

        if not self.camera:
            print("Error: Camera not found!")
            robot.cleanup()
            exit(1)

        self.camera.enable(PioneerControllers.time_step)

        self.lidar: Lidar = robot.getDevice("lidar")

        if not self.lidar:
            print("Error: lidar not found!")
            robot.cleanup()
            exit(1)

        self.lidar.enable(PioneerControllers.time_step)
        self.lidar.enablePointCloud()

    def moveFoward(self):
        self.front_left.setVelocity(self.default_speed)
        self.front_right.setVelocity(self.default_speed)
        self.rear_left.setVelocity(self.default_speed)
        self.rear_right.setVelocity(self.default_speed)

    def moveBackward(self):
        self.front_left.setVelocity(-self.default_speed)
        self.front_right.setVelocity(-self.default_speed)
        self.rear_left.setVelocity(-self.default_speed)
        self.rear_right.setVelocity(-self.default_speed)

    def moveFowardRight(self):
        self.front_left.setVelocity(self.default_speed)
        self.rear_left.setVelocity(self.default_speed)

        self.front_right.setVelocity(self.default_speed * 1.5)
        self.rear_right.setVelocity(self.default_speed * 1.5)

    def moveFowardLeft(self):
        self.front_left.setVelocity(self.default_speed)
        self.rear_left.setVelocity(self.default_speed)

        self.front_right.setVelocity(self.default_speed * 1.5)
        self.rear_right.setVelocity(self.default_speed * 1.5)

    def rotateLeft(self):
        self.front_left.setVelocity(-self.default_speed)
        self.rear_left.setVelocity(-self.default_speed)

        self.front_right.setVelocity(self.default_speed)
        self.rear_right.setVelocity(self.default_speed)

    def rotateRight(self):
        self.front_left.setVelocity(self.default_speed)
        self.rear_left.setVelocity(self.default_speed)

        self.front_right.setVelocity(-self.default_speed)
        self.rear_right.setVelocity(-self.default_speed)

    def stop(self):
        self.front_left.setVelocity(0.0)
        self.rear_left.setVelocity(0.0)

        self.front_right.setVelocity(0.0)
        self.rear_right.setVelocity(0.0)

    def streamImage(self):
        image = self.camera.getImageArray()
        write_matrix_data(image)

        return image

    def lidarData(self):
        lidar_data = self.lidar.getRangeImage()
        lidar_width = self.lidar.getHorizontalResolution()
        lidar_height = self.lidar.getNumberOfLayers()
        lidar_max_range = self.lidar.getMaxRange()
        
        write_lidar_object_data(lidar_data, lidar_width, lidar_height, lidar_max_range)
        return lidar_data
    
    def set_position(self, x, y, theta): # só pode ser usado para treinamento, com o supervisor true e def do robo "Pioneer"
        robot_node = robot.getSelf()
        translation = robot_node.getField("translation")
        
        translation.setSFVec3f([x, y, 0])
        
        rotation = robot_node.getField("rotation")
        rotation.setSFRotation([0, 0, 1, theta])
        
    def wait_step(self):
        """Avança a simulação e espera o passo completar"""
        if robot.step(self.time_step) == -1:
            robot.cleanup()
            exit(0)
    
    def collect_training_data(self, num_samples=1000):
        features_rgb = []
        features_depth = []
        targets_dist = []
        targets_angle = []
        
        rng = np.random.default_rng()
        for i in range(num_samples):
            print(i)
            # posiciona robo aleatoriamente
            x, y = generate_position()
            theta = rng.uniform(0, 2 * np.pi)
            
            self.set_position(x, y, theta)
            
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            self.wait_step()
            
            
            # coleta dados
            rgb = self.streamImage()
            lidar = self.lidarData()
            depth_img = pp.range_img_to_img(lidar, self.lidar.getHorizontalResolution(), self.lidar.getNumberOfLayers(), self.lidar.getMaxRange())
            
            # calcular targets
            dist, angle = calculate_targets(x, y, theta)
            
            
            # armazena dados
            features_rgb.append(rgb)
            features_depth.append(depth_img)
            targets_dist.append(dist)
            targets_angle.append(angle)
            print(np.shape(features_rgb))
            
        np.savez('training_data.npz',
                    rgb=features_rgb,
                    depth=features_depth,
                    dist=targets_dist,
                    angle=targets_angle)
        
        return features_rgb, features_depth, targets_dist, targets_angle
    


# Initialize Robot
pioneer = PioneerControllers()
keyboard = Keyboard()

keyboard.enable(PioneerControllers.time_step)

while robot.step(PioneerControllers.time_step) != -1:
    key = keyboard.getKey()

    if key == -1:
        pioneer.stop()

    if key == ord("W"):
        pioneer.moveFoward()

    if key == ord("S"):
        pioneer.moveBackward()

    if key == ord("A"):
        pioneer.rotateLeft()

    if key == ord("D"):
        pioneer.rotateRight()

    if key == ord("L"):
        lidar = pioneer.lidar
        
        show_lidar_img(pioneer.lidarData(), lidar.getHorizontalResolution(), lidar.getNumberOfLayers(), lidar.getMaxRange())

    if key == ord("C"):
        show_camera_img(pioneer.streamImage())

    if key == ord("K"):
        print("getting camera and lidar sample")
        pioneer.streamImage()
        print("image saved")
        pioneer.lidarData()
        print("lidar saved")
    
    if key == ord("T"):
        pioneer.collect_training_data(1000)