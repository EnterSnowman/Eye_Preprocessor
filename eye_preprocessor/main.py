import argparse
import os
from eye_preprocessor.eye_patches_preprocessor import EyePreprocessor
import cv2 as cv
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-patches-dir", "-patches-dir", type=str, default="patches")
    parser.add_argument("--config_file", "-conf", type=str, default="example_conf.json")
    parser.add_argument("--video-file", "-video", type=str, default="../some_videos/2.avi")
    args = parser.parse_args()
    if not os.path.exists(args.saved_patches_dir):
        os.makedirs(args.saved_patches_dir)
    ep = EyePreprocessor("example_conf.json")

    for patch in ep.get_patches_from_video(args.video_file, cache_dir="../cache", use_cache=False):
        time_stamp = str(time())
        left_name = os.path.join(args.saved_patches_dir,args.saved_patches_dir+time_stamp+"_left.png")
        right_name = os.path.join(args.saved_patches_dir,args.saved_patches_dir+str(time())+"_right.png")
        cv.imwrite(left_name, patch[0])
        cv.imwrite(right_name, patch[1])

