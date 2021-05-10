import io
import math

import cv2
import numpy as np
import pyrealsense2 as rse

from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points_and_skeleton, joints_dict

meter_coordinates = ''
point_coordinates = ''


# Using Euclidean distance formula
def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# def write_to_csv(meter_coordinates, point_coordinates, x0, y0, conf0, x1, y1, conf1):



def calc(x0, y0, conf0, x1, y1, conf1, aligned_depth, frame_intrin, frame_name, depth_scale, is_right_shoulder,
         time_stamp):
    global meter_coordinates
    global point_coordinates
    pixel_1 = [x0, y0]
    pixel_2 = [x1, y1]
    depth_1 = aligned_depth[int(x0), int(y0)]
    depth_2 = aligned_depth[int(x1), int(y1)]
    # depth * depth scale = physical distance between the camera and the baby
    phys_depth_1 = depth_1 * depth_scale
    phys_depth_2 = depth_2 * depth_scale

    # Getting depth using this API return wrong value, no idea why
    # phys_depth_1 = depth_frame.get_distance(pixel_1[0], pixel_1[1])
    # phys_depth_2 = depth_frame.get_distance(pixel_2[0], pixel_2[1])
    # depth_1 = phys_depth_1 / depth_scale
    # depth_2 = phys_depth_2 / depth_scale
    # if it is negative then it is incorrect because it means the baby is behind the camera
    # max limit is the expected furthest distance between the baby and the camera,
    # this might be vary each time we record
    if (-1 < phys_depth_1 < 1.9) and (-1 < phys_depth_2 < 1.9):
        point1 = rse.rs2_deproject_pixel_to_point(frame_intrin, pixel_1, phys_depth_1)
        point2 = rse.rs2_deproject_pixel_to_point(frame_intrin, pixel_2, phys_depth_2)
        # This get the distance in meters, we multiply by 100 to get cm
        length = distance(point1[1], point1[0], point1[2], point2[1], point2[0], point2[2]) * 100
        if frame_name not in meter_coordinates:
            meter_coordinates += '\n' + str(frame_name) + ',' + str(point1[0]) + ',' + str(point1[1]) + ',' \
                                 + str(point1[2]) + ',' + str(conf0) + ',' + str(point2[0]) + ',' + str(point2[1]) \
                                 + ',' + str(point2[2]) + ',' + str(conf1)
        else:
            if is_right_shoulder:
                meter_coordinates += ',' + str(point1[0]) + ',' + str(point1[1]) + ',' + str(point1[2]) + ',' \
                                     + str(conf0) + ',' + str(point2[0]) + ',' + str(point2[1]) + ',' + str(point2[2]) \
                                     + ',' + str(conf1)
            else:
                meter_coordinates += ',' + str(point2[0]) + ',' + str(point2[1]) + ',' + str(point2[2]) + ',' \
                                     + str(conf1)
        if frame_name not in point_coordinates:
            point_coordinates += '\n' + str(frame_name) + ',' + str(y0) + ',' + str(x0) + ',' + str(depth_1) + ',' \
                                 + str(conf0) + ',' + str(y1) + ',' + str(x1) + ',' + str(depth_2) + ',' + str(conf1)
        else:
            if is_right_shoulder:
                point_coordinates += ',' + str(y0) + ',' + str(x0) + ',' + str(depth_1) + ',' + str(conf0) + ',' \
                                     + str(y1) + ',' + str(x1) + ',' + str(depth_2) + ',' + str(conf1)
            else:
                point_coordinates += ',' + str(y1) + ',' + str(x1) + ',' + str(depth_2) + ',' + str(conf1)
            return length
    else:
        return None


def algorithm(model, profile, used_file, pipe):
    global meter_coordinates
    global point_coordinates
    output_vid_name = './output/sub8/5m/' + used_file + '.avi'
    vid_writer = cv2.VideoWriter(output_vid_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

    # Threshold confidence, only use predicted joints with confidence higher than this
    confidence = 0.7
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
    # spatial_filtering.set_option(rse.option.filter_magnitude, 5)
    # spatial_filtering.set_option(rse.option.filter_smooth_alpha, 0.25)
    # spatial_filtering.set_option(rse.option.filter_smooth_delta, 30)
    # spatial_filtering.set_option(rse.option.holes_fill, 5)

    temporal_filtering = rse.temporal_filter()

    # previous_left_wrist = []
    # previous_right_wrist = []
    # distances = ''

    meter_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    meter_coordinates += 'Frame,LS,LS,LS,LS,LE,LE,LE,LE,LW,LW,LW,LW,RS,RS,RS,RS,RE,RE,RE,RE,RW,RW,RW,RW' \
                         + ',LH,LH,LH,LH,RH,RH,RH,RH' + '\n' \
                         + ' ,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf' \
                         + 'x,y,z,conf'

    point_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    point_coordinates += 'Frame,LS,LS,LS,LS,LE,LE,LE,LE,LW,LW,LW,LW,RS,RS,RS,RS,RE,RE,RE,RE,RW,RW,RW,RW' \
                         + ',LH,LH,LH,LH,RH,RH,RH,RH' \
                         + '\n' + ' ,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,x,y,z,conf,' \
                         + 'x,y,z,conf,x,y,z,conf'
    frame_index = 0
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_frames_array = []
    frame_timestamp_array = []
    aligned_depth_frames_array = []
    aligned_frames = None
    intrin_type = None
    try:
        align = rse.align(rse.stream.depth)
        while True:
            # If we have multicam, consider changing to poll_for_frames
            frames = pipe.wait_for_frames(50000)
            playback.pause()
            aligned_frames = align.process(frames)

            # aligned_depth_frame is a 848x480 depth image
            aligned_depth_frame = aligned_frames.get_depth_frame().as_depth_frame()

            color_frame = frames.get_color_frame()
            frame_timestamp = color_frame.get_timestamp()

            if not aligned_depth_frame or not color_frame:
                continue

            if frame_timestamp not in frame_timestamp_array:
                color_frames_array.append(color_frame)
                aligned_depth_frames_array.append(aligned_depth_frame)
                frame_timestamp_array.append(frame_timestamp)
                # if len(color_frames_array) >= 2:
                #     break

            playback.resume()
    except RuntimeError as exception:
        print(exception)
        print("There are no more frames left in the .bag file!")
        print("array: ", len(color_frames_array))
        print("timestamp array: ", len(frame_timestamp_array))
        print("first frame: ", color_frames_array[0].get_timestamp())
        print("second frame: ", color_frames_array[1].get_timestamp())
    finally:
        pass

    print("color array: ", len(color_frames_array))
    print("depth array: ", len(aligned_depth_frames_array))

    try:
        for color_frame in color_frames_array:
            frame_index += 1
            frame_name = 'Frame ' + str(frame_index)
            color = np.asanyarray(color_frame.as_frame().get_data())
            # Color frame is in RGB but openCV is using BGR, so we convert it
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            extracted_frame = color

            # begin of drawing skeleton by HRnet
            # This will predict the whole skeleton, including other joints that we don't focus on
            joints = model.predict(color)

            for index, points in enumerate(joints[1]):
                # Remove unnecessary joints
                for v in range(0, 9):
                    if v < 5:
                        points = np.delete(points, 0, axis=0)
                    else:
                        points = np.delete(points, 8, axis=0)

                color = draw_points_and_skeleton(color, points, joints_dict()["coco"]['skeleton'], person_index=index,
                                                 joints_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                 joints_palette_samples=1)

                print('\nFrame number: ', frame_index)
                # Each joint has 3 values: 0 -> 1 -> 2 (y position, x position, joint confidence)
                print('Left Shoulder: ', points[0, 0], points[0, 1], points[0, 2] * 100)
                print('Right Shoulder: ', points[1, 0], points[1, 1], points[1, 2] * 100)
                print('Left Elbow: ', points[2, 0], points[2, 1], points[2, 2] * 100)
                print('Right Elbow: ', points[3, 0], points[3, 1], points[3, 2] * 100)
                print('Left Wrist: ', points[4, 0], points[4, 1], points[4, 2] * 100)
                print('Right Wrist: ', points[5, 0], points[5, 1], points[5, 2] * 100)

                if index == 0:
                    count = count + 1
                    aligned_depth_frame = aligned_depth_frames_array[frame_index - 1]

                    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
                    aligned_depth_frame = spatial_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = temporal_filtering.process(aligned_depth_frame)
                    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)

                    aligned_depth_frame = aligned_depth_frame.as_depth_frame()
                    aligned_depth = np.asanyarray(aligned_depth_frame.as_frame().get_data())

                    # Depth
                    intrin_type = 'Depth'
                    frame_intrin = profile.get_stream(rse.stream.depth).as_video_stream_profile().intrinsics
                    # Color
                    # intrin_type = 'Color'
                    # frame_intrin = color_frame.profile.as_video_stream_profile().intrinsics

                    # If Algorithm's confidence is higher than confidence threshold then calculate joint's position
                    # based on that point

                    # if (int(round(pt[0, 0], 0)) < p[0]) and (int(round(pt[0, 1], 0)) < p[1])
                    # and (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1]):
                    # if (points[0, 2] > confidence) and (points[2, 2] > confidence):
                    print("Left Shoulder to Left Elbow length: ")
                    length = calc(float(round(points[0, 0], 1)), float(round(points[0, 1], 1)),
                                  points[0, 2], float(round(points[2, 0], 1)),
                                  float(round(points[2, 1], 1)), points[2, 2],
                                  aligned_depth, frame_intrin, frame_name, depth_scale, False, None)

                    if length is not None:
                        ls_le.append(length)
                    # else:
                    #     length = calc(0, 0, points[0, 2], 0, 0, points[2, 2],
                    #                   aligned_depth, frame_intrin, frame_name, depth_scale, False, None)

                    # if (int(round(pt[2, 0], 0)) < p[0]) and (int(round(pt[2, 1], 0)) < p[1])
                    # and (int(round(pt[4, 0], 0)) < p[0]) and (int(round(pt[4, 1], 0)) < p[1]):
                    # if points[4, 2] > confidence:
                    # if points[2, 2] > confidence:
                    length = calc(float(round(points[2, 0], 1)), float(round(points[2, 1], 1)),
                                  points[2, 2], float(round(points[4, 0], 1)),
                                  float(round(points[4, 1], 1)), points[4, 2],
                                  aligned_depth, frame_intrin, frame_name, depth_scale, False, None)

                    if length is not None:
                        le_lw.append(length)
                        # else:
                        #     length = calc(0, 0, points[2, 2], int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                        #                   points[4, 2], aligned_depth, frame_intrin, frame_name, depth_scale, False,
                        #                   None)
                    # else:
                    #     length = calc(0, 0, points[2, 2], int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                    #                   points[4, 2],
                    #                   aligned_depth, frame_intrin, frame_name, depth_scale, False, None)
                        # if len(previous_left_wrist) != 0:
                        #     left_wrist_movement = calc(int(round(points[4, 0], 0)), int(round(points[4, 1], 0)),
                        #                                points[4, 2],
                        #                                int(round(previous_left_wrist[0], 0)),
                        #                                int(round(previous_left_wrist[1], 0)),
                        #                                -1, aligned_depth, frame_intrin, frame_name,
                        #                                depth_scale, aligned_depth_frame, None)
                        #     distances += 'left movement from frame' + str(frame_index - 1) + ' to frame ' \
                        #                  + str(frame_index) + ' ' + str(left_wrist_movement) + 'cm ' + str(
                        #         points[4, 2]) + '\n'
                        # previous_left_wrist = points[4, :]

                    # if (int(round(pt[1, 0], 0) < p[0])) and (int(round(pt[1, 1], 0)) < p[1])
                    # and (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1]):
                    # if (points[1, 2] > confidence) and (points[3, 2] > confidence):
                    print("Right Shoulder to Right Elbow length: ")
                    length = calc(float(round(points[1, 0], 1)), float(round(points[1, 1], 1)),
                                  points[1, 2], float(round(points[3, 0], 1)),
                                  float(round(points[3, 1], 1)), points[3, 2],
                                  aligned_depth, frame_intrin, frame_name, depth_scale, True, None)

                    if length is not None:
                        rs_re.append(length)
                    # else:
                    #     length = calc(0, 0, points[1, 2], 0, 0, points[3, 2],
                    #                   aligned_depth, frame_intrin, frame_name, depth_scale, True, None)

                    # if (int(round(pt[3, 0], 0)) < p[0]) and (int(round(pt[3, 1], 0)) < p[1])
                    # and (int(round(pt[5, 0], 0)) < p[0]) and (int(round(pt[5, 1], 0)) < p[1]):
                    # if points[5, 2] > confidence:
                    #     if points[3, 2] > confidence:
                    print("Right Elbow to Right Wrist length: ")
                    length = calc(float(round(points[3, 0], 1)), float(round(points[3, 1], 1)),
                                  points[3, 2], float(round(points[5, 0], 1)),
                                  float(round(points[5, 1], 1)), points[5, 2],
                                  aligned_depth, frame_intrin, frame_name, depth_scale,
                                  False, frame_timestamp_array[frame_index - 1])

                    if length is not None:
                        re_rw.append(length)

                    length = calc(float(round(points[6, 0], 1)), float(round(points[6, 1], 1)),
                                  points[6, 2], float(round(points[7, 0], 1)),
                                  float(round(points[7, 1], 1)), points[7, 2],
                                  aligned_depth, frame_intrin, frame_name, depth_scale, True, None)

                    ls = ls + points[0, 2]
                    rs = rs + points[1, 2]
                    le = le + points[2, 2]
                    re = re + points[3, 2]
                    lw = lw + points[4, 2]
                    rw = rw + points[5, 2]


                    #     else:
                    #         length = calc(0, 0, points[3, 2], int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                    #                       points[5, 2],
                    #                       aligned_depth, frame_intrin, frame_name, depth_scale,
                    #                       False, frame_timestamp_array[frame_index - 1])
                    # else:
                    #     length = calc(0, 0, points[3, 2], 0, 0, points[5, 2],
                    #                   aligned_depth, frame_intrin, frame_name, depth_scale,
                    #                   False, frame_timestamp_array[frame_index - 1])

                        # if len(previous_right_wrist) != 0:
                        #     right_wrist_movement = calc(int(round(points[5, 0], 0)), int(round(points[5, 1], 0)),
                        #                                 points[5, 2],
                        #                                 int(round(previous_right_wrist[0], 0)),
                        #                                 int(round(previous_right_wrist[1], 0)),
                        #                                 -1, aligned_depth,
                        #                                 frame_intrin, frame_name, depth_scale, False,
                        #                                 None)
                            # distances += 'right movement from frame ' + str(frame_index - 1) + ' to frame ' \
                            #              + str(frame_index) + ' ' + str(right_wrist_movement) + 'cm ' + str(
                            #     points[5, 2]) + '\n'
                # previous_right_wrist = points[5, :]
                # print('previous right wrist: ', previous_right_wrist)

                        # end of drawing skeleton by HRnet

            cv2.imshow('Output Frame', color)
            test_directory = "./output/sub8/5m/png/"
            file_path = test_directory + 'savedImage' + str(frame_index) + '.png'
            cv2.imwrite(file_path, color)
            cv2.waitKey(1) & 0xFF
            vid_writer.write(color)
    except RuntimeError as exception:
        print(exception)
    finally:
        print('Left Shoulder to Elbow: ', round(np.mean(ls_le), 4))
        print('Left Elbow to Wrist: ', round(np.mean(le_lw), 4))
        print('Right Shoulder to Elbow: ', round(np.mean(rs_re), 4))
        print('Right Elbow to Wrist: ', round(np.mean(re_rw), 4))

        summary_report = '\n' + 'Total processed frames: ' + str(frame_index) \
            + '\n' + 'Left Shoulder to Elbow: ' + str(round(np.mean(ls_le), 4)) \
            + '\n' + 'Left Elbow to Wrist: ' + str(round(np.mean(le_lw), 4)) \
            + '\n' + 'Right Shoulder to Elbow: ' + str(round(np.mean(rs_re), 4)) \
            + '\n' + 'Right Elbow to Wrist: ' + str(round(np.mean(re_rw), 4)) + '\n'
        meter_coordinates = summary_report + meter_coordinates
        point_coordinates = summary_report + point_coordinates

        s = io.StringIO(meter_coordinates)
        path = './output/sub8/5m/data/' + used_file
        with open(path + "_" + intrin_type + '_co_meter_color_intrin.csv', 'w') as f:
            for line in s:
                f.write(line)
            f.close()
        point_coordinates += '\n' + 'Total processed frames: ' + str(frame_index)
        s = io.StringIO(point_coordinates)
        with open(path + "_" + intrin_type + '_co_point_color_intrin_09.csv', 'w') as f:
            for line in s:
                f.write(line)
            f.close()
        pipe.stop()
        pass


if __name__ == '__main__':
    # Init SimpleHRNet library
    model_setup = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")

    vid_file = 'sub8_5m_spon'
    video = './input/sub8/5m/' + vid_file + '.bag'

    # Construct a pipeline which abstracts the device
    # noinspection PyArgumentList
    pipeline = rse.pipeline()
    # Create a configuration for configuring the pipeline with a non default profile
    cfg = rse.config()
    cfg.enable_stream(rse.stream.depth, 640, 480, rse.format.z16, 30)
    cfg.enable_stream(rse.stream.color, 640, 480, rse.format.rgb8, 30)
    cfg.enable_device_from_file(video, repeat_playback=False)
    # Instruct pipeline to start streaming with the requested configuration
    pipeline_profile = pipeline.start(cfg)
    playback = pipeline_profile.get_device().as_playback()
    playback.set_real_time(False)

    algorithm(model_setup, pipeline_profile, vid_file, pipeline)
