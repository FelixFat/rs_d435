import sys
import csv
import time

import pyrealsense2 as rs
import numpy as np
import cv2


IMG_SHAPE = (640, 480) # Input image resolution
FPS_FRAME = 30 # fps
FREQ_FRAME = 0 # s
MAX_DIST = 4.0 # m
DEPTH_SCALE = 0.0010000000474974513


def get_localtime():
    return time.strftime('%d%m%Y_%H%M%S', time.localtime())


def make_csv(path: str):
    with open(path, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['Local Time', 'HaG'])


def write_csv(path: str, num: int, height: float):
    with open(path, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([num, height])


def read_im(im_path: str, im_type: int) -> np.ndarray:
    im = cv2.imread(im_path, im_type)
    return im


def write_im(save_path: str, im: np.ndarray):
    cv2.imwrite(f'{save_path}', im)
    return f"Saved to '{save_path}'"


def rs_data2dmap(data: np.ndarray, clip_dist: float) -> np.ndarray:
    return np.where((data > clip_dist), 0, data / clip_dist * 255)


def rs_dmap2data(dmap: np.ndarray) -> np.ndarray:
    return dmap * MAX_DIST / 255


def rs_data2pc(data: np.ndarray):
    pc = rs.pointcloud()
    pointcloud = pc.calculate(data)
    return np.asanyarray(pointcloud.get_vertices())


def video_stream():
    d_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    clip_dist = MAX_DIST / d_scale

    localtime = get_localtime()
    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()

        if not depth or not color:
            print(">> ERROR: NO FRAME DATA!")
            continue
        
        wc, hc = color.get_width(), color.get_height()
        wd, hd = depth.get_width(), depth.get_height()
        dist = depth.get_distance(wd // 2, hd // 2)

        print('Color:', wc, hc)
        print('Depth:', wd, hd, dist)

        im_color = np.asanyarray(color.get_data())
        im_depth = np.asanyarray(depth.get_data())

        images = np.hstack((
            cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=0.03), cv2.COLORMAP_JET),
            cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        ))

        if '-r' in sys.argv:
            write_csv('./records/log.csv', localtime, dist)
            print(write_im(f"./records/color/color_{localtime}.jpg", cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)))
            print(write_im(f"./records/depth/depth_{localtime}.jpg", rs_data2dmap(im_depth, clip_dist)))

        cv2.namedWindow('RS D435 stream', cv2.WINDOW_NORMAL)
        cv2.imshow('RS D435 stream', images)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return
        else:
            '''
            depth_intrinsic =  rs.intrinsics()
            depth_intrinsic.width = 640
            depth_intrinsic.height = 480
            depth_intrinsic.ppx = 213.47621154785156
            depth_intrinsic.ppy = 121.29695892333984
            depth_intrinsic.fx = 306.0126953125
            depth_intrinsic.fy = 306.1602783203125
            depth_intrinsic.model = rs.pyrealsense2.distortion.inverse_brown_conrady
            depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [wd // 2, hd // 2], dist)
            print(depth_point)

            pointcloud = rs_data2pc(depth)
            mid = len(pointcloud) // 2
            print(pointcloud[mid])
            '''
            
            print("-" * 40)
            time.sleep(FREQ_FRAME)


def show_test():
    print(get_localtime())

    dmap = read_im('./include/test_depth.jpg', cv2.IMREAD_GRAYSCALE)
    depth = rs_dmap2data(dmap)

    color = read_im('./include/test_color.jpg', cv2.IMREAD_COLOR)

    print(
        dmap[dmap.shape[0] // 2, dmap.shape[1] // 2],
        depth[depth.shape[0] // 2, depth.shape[1] // 2]
    )

    while True:
        images = np.hstack((
            cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET),
            cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        ))

        cv2.imshow('', images)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return


def main():
    global pipe
    global profile
    global align

    if '-s' in sys.argv:
        pipe = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, IMG_SHAPE[0], IMG_SHAPE[1], rs.format.z16, FPS_FRAME)
        config.enable_stream(rs.stream.color, IMG_SHAPE[0], IMG_SHAPE[1], rs.format.rgb8, FPS_FRAME)

        profile = pipe.start(config)
        align = rs.align(rs.stream.color)
        video_stream()

    else:
        show_test()


if __name__ == '__main__':
    main()

