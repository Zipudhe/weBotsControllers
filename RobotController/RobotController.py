from controller import Robot, Keyboard

robot = Robot()
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

        # self.lidar = robot.getDevice("lidar")
        self.camera = robot.getDevice("camera")
        self.camera_width = self.camera.getWidth()

        if not self.camera:
            print("Error: Camera not found!")
            robot.cleanup()
            exit(1)

        self.camera.enable(PioneerControllers.time_step)

        self.lidar = robot.getDevice("lidar")

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

        return image

    def lidarData(self):
        return self.lidar.getPointCloud()


# Initialize Robot
pioneer = PioneerControllers()
keyboard = Keyboard()

keyboard.enable(PioneerControllers.time_step)

while robot.step(PioneerControllers.time_step) != -1:
    key = keyboard.getKey()
    print("keystroke", key)

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
        print("lidar data", pioneer.lidarData())

    if key == ord("C"):
        print("camera image", pioneer.streamImage())
