import csv


def write_matrix_data(matrix):
    fout = open("../matrix_data.txt", "a")
    fout.write(str(matrix) + "\n")
    fout.close()


def write_lidar_object_data(lidar_data):
    flat_lidar_points = [
        coord for point in lidar_data for coord in (point.x, point.y, point.z)
    ]

    with open("../lidar_data.csv", "a") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(flat_lidar_points)
