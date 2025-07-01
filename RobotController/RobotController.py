from controller import Robot, Camera, Lidar

robot = Robot()
camera = Camera("camera")
lidar = Lidar("lidar")

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

    def moveFoward(self):
        self.front_left.setVelocity(self.default_speed)
        self.front_right.setVelocity(self.default_speed)
        self.rear_left.setVelocity(self.default_speed)
        self.rear_right.setVelocity(self.default_speed)

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


# InitializeMotors
pioneer = PioneerControllers()


while robot.step(PioneerControllers.time_step) != -1:
    pioneer.rotateRight()


# devices = []
# qtd_devices = robot.getNumberOfDevices()

# for i in range(qtd_devices):
#    devices.append(robot.getDeviceByIndex(i).getName())
# print("devices", devices)

