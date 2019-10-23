import pyrealsense2 as rse

from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points_and_skeleton, joints_dict, draw_points

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def distance(x1, y1, x2, y2, z1, z2):
    gap = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return gap


def calc(x0, y0, x1, y1, aligned_depth, depth_aligned_intrin, depth_scale):
    pixel_x = [x0, y0]
    pixel_y = [x1, y1]
    depth_x = aligned_depth[x0, y0]
    depth_y = aligned_depth[x1, y1]
    if (0.0 < depth_x*depth_scale < 0.8) and (0.0 < depth_y*depth_scale < 0.8):
        x = rse.rs2_deproject_pixel_to_point(depth_aligned_intrin, pixel_x, depth_x*depth_scale)
        y = rse.rs2_deproject_pixel_to_point(depth_aligned_intrin, pixel_y, depth_y*depth_scale)
        print(depth_x * depth_scale, x[0], x[1], x[2])
        print("Length: ", distance(x[0], y[0], x[1], y[1], x[2], y[2]) * 100)
        return distance(x[0], y[0], x[1], y[1], x[2], y[2]) * 100
    else:
        return None


def algorithm(model, profile):
    vid_writer = cv2.VideoWriter('./outcome/cute_baby02.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    confidence = 0.5
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

    try:
        frame_index = 1
        while True:
            # if we have multicam, consider changing to poll_for_frames
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            color = np.asanyarray(color_frame.get_data())

            # Color frame is in RGB but openCV is using BGR, so we convert it
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

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

                print('\nFrame number: ', frame_index)
                frame_index += 1

                # Each joint has 3 values: 2 -> 1 -> 0 (joint confidence, y position, x position)
                print('Left Shoulder: ', points[0, 2] * 100, points[0, 1], points[0, 0])
                print('Right Shoulder: ', points[1, 2] * 100, points[1, 1], points[1, 0])
                print('Left Elbow: ', points[2, 2] * 100, points[2, 1], points[2, 0])
                print('Right Elbow: ', points[3, 2] * 100, points[3, 1], points[3, 0])
                print('Left Wrist: ', points[4, 2] * 100, points[4, 1], points[4, 0])
                print('Right Wrist: ', points[5, 2] * 100, points[5, 1], points[5, 0])

                if index == 0:
                    count = count + 1

                    alignFs = rse.align(rse.stream.depth)
                    frame_align = alignFs.process(frames)

                    aligned_depth_frame = frame_align.get_depth_frame()

                    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                    aligned_depth_frame = spatial_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = temporal_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)

                    aligned_depth = np.asanyarray(aligned_depth_frame.get_data())

                    depth_sensor = profile.get_device().first_depth_sensor()
                    depth_scale = depth_sensor.get_depth_scale()
                    depth_aligned_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

                    # If Algorithm's confidence is higher than confidence threshold then calculate joint's position
                    # based on that point

                    # if (int(round(pt[0, 0], 0)) < p[0]) and (int(round(pt[0, 1], 0)) < p[1])
                    # and (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1]):
                    if (points[0, 2] > confidence) and (points[2, 2] > confidence):
                        print("Left Shoulder to Left Elbow length: ")
                        length = calc(int(round(points[0, 0], 0)), int(round(points[0, 1], 0)),
                                      int(round(points[2, 0], 0)), int(round(points[2, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale)


                        if (length != None):
                            ls_le.append(length)

                    # if (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1])
                    # and (int(round(pt[4, 0], 0)) < p[0]) and (int(round(pt[4, 1], 0)) < p[1]):
                    if (points[2, 2] > confidence) and (points[4, 2] > confidence):
                        length = calc(int(round(points[2, 0], 0)), int(round(points[2, 1], 0)),
                                      int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale)

                        if (length != None):
                            le_lw.append(length)

                    if points[4, 2] > confidence:
                        if len(previous_left_wrist) != 0:
                            left_wrist_movement = calc(int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                                                       int(round(previous_left_wrist[0], 0)), int(round(points[0, 1], 0)),
                                                       aligned_depth, depth_aligned_intrin, depth_scale)
                            distances += 'left movement from frame' + str(frame_index - 1) + ' to frame ' \
                                         + str(frame_index) + ' ' + str(left_wrist_movement) + 'cm ' + str(points[4, 2]) + '\n'
                            print('previous left wrist1: ', previous_left_wrist)
                        previous_left_wrist = points[4, :]
                        print('previous left wrist0: ', previous_left_wrist)

                    # if (int(round(pt[1, 0], 0) < p[0])) and (int(round(pt[1, 1], 0)) < p[1])
                    # and (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1]):
                    if (points[1, 2] > confidence) and (points[3, 2] > confidence):
                        print("Right Shoulder to Right Elbow length: ")
                        length = calc(int(round(points[1, 0], 0)), int(round(points[1, 1], 0)),
                                      int(round(points[3, 0], 0)), int(round(points[3, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale)

                        if (length != None):
                            rs_re.append(length)

                    # if (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1])
                    # and (int(round(pt[5, 0], 0)) < p[0]) and (int(round(pt[5, 1], 0)) < p[1]):
                    if (points[3, 2] > confidence) and (points[5, 2] > confidence):
                        print("Right Elbow to Right Wrist length: ")
                        length = calc(int(round(points[3, 0], 0)), int(round(points[3, 1], 0)),
                                      int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                                      aligned_depth, depth_aligned_intrin, depth_scale)

                        if (length != None):
                            re_rw.append(length)

                        ls = ls + points[0, 2]
                        rs = rs + points[1, 2]
                        le = le + points[2, 2]
                        re = re + points[3, 2]
                        lw = lw + points[4, 2]
                        rw = rw + points[5, 2]

                    if points[5, 2] > confidence:
                        if len(previous_right_wrist) != 0:
                            right_wrist_movement = calc(int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                                                       int(round(previous_right_wrist[0], 0)),
                                                       int(round(points[0, 1], 0)),
                                                       aligned_depth, depth_aligned_intrin, depth_scale)
                            distances += 'right movement from frame' + str(frame_index - 1) + ' to frame ' \
                                         + str(frame_index) + ' ' + str(right_wrist_movement) + 'cm ' + str(points[5, 2]) + '\n'
                        previous_right_wrist = points[5, :]
                        print('previous right wrist: ', previous_right_wrist)

            cv2.imshow('Output Frame', color)
            cv2.waitKey(1) & 0xFF
            vid_writer.write(color)
    finally:
        pipe.stop()

        vid_writer.release()
        mean = (ls + rs + le + re + lw + rw) / (count * 6) * 100
        print('\nMean: ', mean)
        print('\nLeft Shoulder: ', ls / count * 100)
        print('\nRight Shoulder: ', rs / count * 100)
        print('\nLeft Elbow: ', le / count * 100)
        print('\nRight Elbow: ', re / count * 100)
        print('\nLeft Wrist: ', lw / count * 100)
        print('\nRight Wrist: ', rw / count * 100)

        print('_________________________')
        print('Left Shoulder to Elbow: ', round(np.mean(ls_le), 4))
        print('Left Elbow to Wrist: ', round(np.mean(le_lw), 4))
        print('Right Shoulder to Elbow: ', round(np.mean(rs_re), 4))
        print('Right Elbow to Wrist: ', round(np.mean(re_rw), 4))

        distance_file = open("./outcome/distance.txt", "w+")
        distance_file.write(distances)
        distance_file.close()

        # plt.plot(rs_re)
        # plt.show()
        pass


if __name__ == '__main__':
    model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
 
    video = './vid/20190912_115237.bag'

    # Construct a pipeline which abstracts the device
    pipe = rse.pipeline()
    # Create a configuration for configuring the pipeline with a non default profile
    cfg = rse.config()
    cfg.enable_device_from_file(video, False)
    # Instruct pipeline to start streaming with the requested configuration
    profile = pipe.start(cfg)

    algorithm(model, profile)
