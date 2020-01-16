import io
import math

import cv2
import numpy as np
import pyrealsense2 as rse

from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points_and_skeleton, joints_dict

coordinates = ''


# Using Euclidean distance formula
def distance(x1, y1, x2, y2, z1, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calc(x0, y0, x1, y1, aligned_depth, depth_aligned_intrin, depth_scale, frame_name, write_to_file):
    global coordinates
    pixel_1 = [x0, y0]
    pixel_2 = [x1, y1]
    depth_1 = aligned_depth[x0, y0]
    depth_2 = aligned_depth[x1, y1]
    phys_depth_1 = depth_1 * depth_scale
    phys_depth_2 = depth_2 * depth_scale
    # depth * depth scale = physical distance between the camera and the baby
    # if it is negative then it is incorrect because it means the baby is behind the camera
    # 1 is the expected furthest distance between the baby and the camera, this might be vary each time we record
    if (0.0 < phys_depth_1 < 1) and (0.0 < phys_depth_2 < 1):
        point1 = rse.rs2_deproject_pixel_to_point(depth_aligned_intrin, pixel_1, phys_depth_1)
        point2 = rse.rs2_deproject_pixel_to_point(depth_aligned_intrin, pixel_2, phys_depth_2)
        # This get the distance in meters, we multiply by 100 to get cm
        length = distance(point1[0], point1[1], point2[0], point2[1], point1[2], point2[2]) * 100
        if write_to_file:
            print("get to this: ", frame_name)
            if frame_name not in coordinates:
                coordinates += '\n' + str(frame_name) + ',' + str(point1[0]) + ',' + str(point1[1]) + ',' \
                               + str(point1[2]) + ',' + str(point2[0]) + ',' + str(point2[1]) + ',' + str(point2[2])
            else:
                coordinates += ',' + str(point2[0]) + ',' + str(point2[1]) + ',' + str(point2[2])
        return length
    else:
        return None


def algorithm(model, profile):
    global coordinates
    vid_writer = cv2.VideoWriter('./outcome/cute_baby02.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    # Threshold confidence, only use predicted joints with confidence higher than this
    confidence = 0.3
    count = 1
    # Left Shoulder
    ls = 0
    # Right Shoulder
    rs = 0
    # Left Elbow
    le = 0
    # Right Elbow
    re = 0
    # Left Wrist
    lw = 0
    # Right Wrist
    rw = 0

    # init 4 empty arrays to store lengths from each shoulder to elbow and from elbow to wrist
    ls_le, rs_re, le_lw, re_rw = ([] for i in range(4))

    decimation_filtering = rse.decimation_filter()
    decimation_filtering.set_option(rse.option.filter_magnitude, 2)

    depth_to_disparity = rse.disparity_transform(True)
    disparity_to_depth = rse.disparity_transform(False)

    # Spatial filter is used for smoothing the image
    spatial_filtering = rse.spatial_filter()
    spatial_filtering.set_option(rse.option.filter_magnitude, 5)
    spatial_filtering.set_option(rse.option.filter_smooth_alpha, 0.25)
    spatial_filtering.set_option(rse.option.filter_smooth_delta, 30)
    spatial_filtering.set_option(rse.option.holes_fill, 5)

    temporal_filtering = rse.temporal_filter()

    previous_left_wrist = []
    previous_right_wrist = []
    distances = ''

    coordinates += '\n' + 'Processed File : 20191030_134157.bag' + '\n'
    coordinates += 'Frame,RS,RS,RS,RE,RE,RE,RW,RW,RW' + '\n' + ' ,x,y,z,x,y,z,x,y,z'
    frame_index = 0
    align_fs = rse.align(rse.stream.depth)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    frames_array = []
    frame_timestamp_array = []
    try:
        while True:
            # If we have multicam, consider changing to poll_for_frames
            frames = pipe.wait_for_frames()
            playback.pause()

            color_frame = frames.get_color_frame()
            frame_timestamp = color_frame.get_timestamp()
            if frame_timestamp not in frame_timestamp_array:
                frames_array.append(color_frame)
                frame_timestamp_array.append(frame_timestamp)

            playback.resume()

    except RuntimeError:

        print("There are no more frames left in the .bag file!")

        print("array: ", len(frames_array))
        print("timestamp array: ", len(frame_timestamp_array))
        print("first frame: ", frames_array[0].get_timestamp())
        print("second frame: ", frames_array[1].get_timestamp())

    finally:
        pass

    try:
        for color_frame in frames_array:
            frame_index += 1
            frame_name = 'Frame ' + str(frame_index)
            color = np.asanyarray(color_frame.get_data())
            # Color frame is in RGB but openCV is using BGR, so we convert it
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # begin of drawing skeleton by HRnet
            # This will predict the whole skeleton, including other joints that we don't focus on
            joints = model.predict(color)

            for index, points in enumerate(joints):
                # Remove unnecessary joints
                for v in range(0, 11):
                    if v < 5:
                        points = np.delete(points, 0, axis=0)
                    else:
                        points = np.delete(points, 6, axis=0)

                color = draw_points_and_skeleton(color, points, joints_dict()["coco"]['skeleton'], person_index=index,
                                                 joints_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                 joints_palette_samples=1)

                # print('\nFrame number: ', frame_index)

                # Each joint has 3 values: 0 -> 1 -> 2 (x position, y position, joint confidence)
                print('Left Shoulder: ', points[0, 0], points[0, 1], points[0, 2] * 100)
                print('Right Shoulder: ', points[1, 0], points[1, 1], points[1, 2] * 100)
                print('Left Elbow: ', points[2, 0], points[2, 1], points[2, 2] * 100)
                print('Right Elbow: ', points[3, 0], points[3, 1], points[3, 2] * 100)
                print('Left Wrist: ', points[4, 0], points[4, 1], points[4, 2] * 100)
                print('Right Wrist: ', points[5, 0], points[5, 1], points[5, 2] * 100)

                if index == 0:
                    count = count + 1

                    frame_align = align_fs.process(frames)

                    aligned_depth_frame = frame_align.get_depth_frame()

                    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                    aligned_depth_frame = spatial_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = temporal_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)

                    aligned_depth = np.asanyarray(aligned_depth_frame.get_data())

                    depth_aligned_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

                    # If Algorithm's confidence is higher than confidence threshold then calculate joint's position
                    # based on that point

                    # if (int(round(pt[0, 0], 0)) < p[0]) and (int(round(pt[0, 1], 0)) < p[1])
                    # and (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1]):
                    if (points[0, 2] > confidence) and (points[2, 2] > confidence):
                        # print("Left Shoulder to Left Elbow length: ")
                        length = calc(int(round(points[0, 0], 0)), int(round(points[0, 1], 0)),
                                      int(round(points[2, 0], 0)), int(round(points[2, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale, '', False)

                        if (length != None):
                            ls_le.append(length)

                    # if (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1])
                    # and (int(round(pt[4, 0], 0)) < p[0]) and (int(round(pt[4, 1], 0)) < p[1]):
                    if points[4, 2] > confidence:
                        if points[2, 2] > confidence:
                            length = calc(int(round(points[2, 0], 0)), int(round(points[2, 1], 0)),
                                          int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                                          aligned_depth, depth_aligned_intrin, depth_scale, '', False)

                            if (length != None):
                                le_lw.append(length)

                        if len(previous_left_wrist) != 0:
                            left_wrist_movement = calc(int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                                                       int(round(previous_left_wrist[0], 0)),
                                                       int(round(previous_left_wrist[1], 0)),
                                                       aligned_depth, depth_aligned_intrin, depth_scale, '', False)
                            distances += 'left movement from frame' + str(frame_index - 1) + ' to frame ' \
                                         + str(frame_index) + ' ' + str(left_wrist_movement) + 'cm ' + str(
                                points[4, 2]) + '\n'
                        previous_left_wrist = points[4, :]

                    # if (int(round(pt[1, 0], 0) < p[0])) and (int(round(pt[1, 1], 0)) < p[1])
                    # and (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1]):
                    if (points[1, 2] > confidence) and (points[3, 2] > confidence):
                        # print("Right Shoulder to Right Elbow length: ")
                        length = calc(int(round(points[1, 0], 0)), int(round(points[1, 1], 0)),
                                      int(round(points[3, 0], 0)), int(round(points[3, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale, frame_name, True)

                        if (length != None):
                            rs_re.append(length)

                    # if (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1])
                    # and (int(round(pt[5, 0], 0)) < p[0]) and (int(round(pt[5, 1], 0)) < p[1]):
                    if points[5, 2] > confidence:
                        if points[3, 2] > confidence:
                            # print("Right Elbow to Right Wrist length: ")
                            length = calc(int(round(points[3, 0], 0)), int(round(points[3, 1], 0)),
                                          int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                                          aligned_depth, depth_aligned_intrin, depth_scale, frame_name,
                                          True)

                            if (length != None):
                                re_rw.append(length)

                            ls = ls + points[0, 2]
                            rs = rs + points[1, 2]
                            le = le + points[2, 2]
                            re = re + points[3, 2]
                            lw = lw + points[4, 2]
                            rw = rw + points[5, 2]

                        if len(previous_right_wrist) != 0:
                            right_wrist_movement = calc(int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                                                        int(round(previous_right_wrist[0], 0)),
                                                        int(round(previous_right_wrist[1], 0)),
                                                        aligned_depth, depth_aligned_intrin, depth_scale, '', False)
                            distances += 'right movement from frame ' + str(frame_index - 1) + ' to frame ' \
                                         + str(frame_index) + ' ' + str(right_wrist_movement) + 'cm ' + str(
                                points[5, 2]) + '\n'
                        previous_right_wrist = points[5, :]
                        print('previous right wrist: ', previous_right_wrist)

                        # end of drawing skeleton by HRnet

            cv2.imshow('Output Frame', color)
            cv2.waitKey(1) & 0xFF
            vid_writer.write(color)
    except RuntimeError:
        print("There are no more frames left in the .bag file! 2")
    finally:
        coordinates_file = open("./outcome/coordinators3_conf03_2.txt", "w+")
        coordinates_file.write(coordinates)
        coordinates_file.close()

        coordinates += '\n' + 'Total processed frames: ' + str(frame_index)

        s = io.StringIO(coordinates)
        with open('./outcome/coordinators3_conf03.csv', 'w') as f:
            for line in s:
                f.write(line)
            f.close()
        pass


if __name__ == '__main__':
    # Init SimpleHRNet library
    model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")

    video = './vid/20191030_134157.bag'

    # Construct a pipeline which abstracts the device
    pipe = rse.pipeline()
    # Create a configuration for configuring the pipeline with a non default profile
    cfg = rse.config()
    cfg.enable_device_from_file(video, repeat_playback=False)
    # Instruct pipeline to start streaming with the requested configuration
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    algorithm(model, profile)
