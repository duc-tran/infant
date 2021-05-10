import io
import math

import cv2
import numpy as np
import pyrealsense2 as rse

from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points_and_skeleton, joints_dict

meter_coordinates = ''
point_coordinates = ''


def algorithm(profile, used_file, avi):
    global meter_coordinates
    global point_coordinates

    count = 1

    meter_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    meter_coordinates += 'Frame,LeftShoulder,LS,LS,LE,LE,LE,LW,LW,LW,RS,RS,RS,RE,RE,RE,RW,RW,RW' + '\n' \
                         + ' ,x,y,z,x,y,z,x,y,z' + ',x,y,z,x,y,z,x,y,z'
    point_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    point_coordinates += 'Frame,LS,LS,LS,LE,LE,LE,LW,LW,LW,RS,RS,RS,RE,RE,RE,RW,RW,RW' + '\n' + ' ,x,y,z,x,y,z,x,y,z' \
        + ',x,y,z,x,y,z,x,y,z'

    frame_index = 0
    color_frames_array = []

    vidcap = cv2.VideoCapture(avi)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        color_frames_array.append(image)
        if cv2.waitKey(10) == 27:
            break
        count += 1

    print("color array: ", len(color_frames_array))
    output_vid_name = './output/avi/sub5/5m/' + used_file + '.avi'
    vid_writer = cv2.VideoWriter(output_vid_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    test_directory = "./output/avi/sub7/4m/png/"
    path = './output/avi/sub7/4m/data/' + used_file
    try:
        for color_frame in color_frames_array:
            frame_index += 1
            frame_name = 'Frame ' + str(frame_index)
            color = np.asanyarray(color_frame)
            # Color frame is in RGB but openCV is using BGR, so we convert it
            # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # extracted_frame = color

            # begin of drawing skeleton by HRnet
            # This will predict the whole skeleton, including other joints that we don't focus on
            joints = model.predict(color)

            for index, points in enumerate(joints[1]):
                # Remove unnecessary joints
                for v in range(0, 11):
                    if v < 5:
                        points = np.delete(points, 0, axis=0)
                    else:
                        points = np.delete(points, 6, axis=0)

                color = draw_points_and_skeleton(color, points, joints_dict()["coco"]['skeleton'], person_index=index,
                                                 joints_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                 joints_palette_samples=1)

                if frame_name not in meter_coordinates:
                    point_coordinates += '\n' + str(frame_name) + ',' + str(points[0, 0]) + ',' + str(points[0, 1]) +  \
                                         ',' + str(points[0, 2]) + ',' + str(points[1, 0]) + ',' + str(points[1, 1]) \
                                         + ',' + str(points[1, 2]) + ',' + str(points[2, 0]) + ',' + str(points[2, 1]) \
                                         + ',' + str(points[2, 2]) + ',' + str(points[3, 0]) + ',' + str(points[3, 1]) \
                                         + ',' + str(points[3, 2]) + ',' + str(points[4, 0]) + ',' + str(points[4, 1]) \
                                         + ',' + str(points[4, 2]) + ',' + str(points[5, 0]) + ',' + str(points[5, 1]) \
                                         + ',' + str(points[5, 2])
                cv2.imshow('Output Frame', color)
                file_path = test_directory + 'savedImage' + str(frame_index) + '.png'
                cv2.imwrite(file_path, color)
                cv2.waitKey(1) & 0xFF
                vid_writer.write(color)
    except RuntimeError as exception:
        print(exception)
    except TypeError as error_exception:
        print(error_exception)
    except ValueError as value_exception:
        print(value_exception)
    finally:
        point_coordinates += '\n' + 'Total processed frames: ' + str(frame_index)
        s = io.StringIO(point_coordinates)
        with open(path + "_" + '_co_point_color_intrin_09.csv', 'w') as f:
            for line in s:
                f.write(line)
            f.close()
        pass


if __name__ == '__main__':
    # Init SimpleHRNet library
    model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")

    used_avi = 'sub7_4m_spon'
    avi = './input/avi/sub7/' + used_avi + '.avi'

    algorithm(model, used_avi, avi)
