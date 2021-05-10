import io

import numpy as np
import pyrealsense2 as rse
import csv

from SimpleHRNet import SimpleHRNet

meter_coordinates = ''
point_coordinates = ''


def algorithm(profile, used_file, pipe):
    global meter_coordinates
    global point_coordinates

    count = 1

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

    meter_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    meter_coordinates += 'Frame,LS,LS,LS,LE,LE,LE,LW,LW,LW,RS,RS,RS,RE,RE,RE,RW,RW,RW' + '\n' + ' ,x,y,z,x,y,z,x,y,z' \
        + ',x,y,z,x,y,z,x,y,z'
    point_coordinates += '\n' + 'Processed File : ' + used_file + '\n'
    point_coordinates += 'Frame,LS,LS,LS,LE,LE,LE,LW,LW,LW,RS,RS,RS,RE,RE,RE,RW,RW,RW' + '\n' + ' ,x,y,z,x,y,z,x,y,z' \
        + ',x,y,z,x,y,z,x,y,z'

    frame_index = 0
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    color_frames_array = []
    frame_timestamp_array = []
    aligned_depth_frames_array = []
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

    tt_coordinates = []
    with open('./test_csv/pos178.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tt_coordinates.append(row)

    try:
        for color_frame in color_frames_array:
            frame_index += 1
            frame_name = 'Frame ' + str(frame_index)
            color = np.asanyarray(color_frame.as_frame().get_data())
            # Color frame is in RGB but openCV is using BGR, so we convert it

            # begin of drawing skeleton by HRnet
            # This will predict the whole skeleton, including other joints that we don't focus on

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

            if frame_index >= 2:
                x0 = round(int(float(tt_coordinates[frame_index-1][0])), 0)
                y0 = round(int(float(tt_coordinates[frame_index-1][1])), 0)
                pixel = [x0, y0]
                depth = aligned_depth[y0, x0]
                phys_depth = depth * depth_scale
                meter_point = rse.rs2_deproject_pixel_to_point(frame_intrin, pixel, phys_depth)
                meter_coordinates += '\n' + str(frame_name) + ',' + str(meter_point[0]) + ',' + str(meter_point[1]) \
                                     + ',' + str(meter_point[2])
                point_coordinates += '\n' + str(frame_name) + ',' + str(x0) + ',' + str(y0) + ',' \
                                     + str(depth)

            test_directory = "./avi/turn_table/"
    except RuntimeError as exception:
        print(exception)
    finally:
        s = io.StringIO(meter_coordinates)
        path = './avi/turn_table/' + used_file
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
    model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")

    used_file = 'TurntableTest2AE'
    video = './input/turn_table/' + used_file + '.bag'

    # Construct a pipeline which abstracts the device
    pipe = rse.pipeline()
    # Create a configuration for configuring the pipeline with a non default profile
    cfg = rse.config()
    cfg.enable_stream(rse.stream.depth, 640, 480, rse.format.z16, 30)
    cfg.enable_stream(rse.stream.color, 640, 480, rse.format.rgb8, 30)
    cfg.enable_device_from_file(video, repeat_playback=False)
    # Instruct pipeline to start streaming with the requested configuration
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    algorithm(profile, used_file, pipe)
