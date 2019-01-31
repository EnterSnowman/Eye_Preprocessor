def get_argument_parser():
    import argparse
    config_file_help = """
    Path to eye preprocessor configuration file
    """
    video_folder_help = """
    Path to folder with videos, which will be proceeded by eye preprocessor
    """
    use_cache_help = """
    Using and creating cache for videos in given folder
    """
    description = """
    Eye preprocessor designed for eye patches extracting from videos, using dlib and opencv.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-file", type=str, help=config_file_help, default="eyepreprocessor\example_conf.json")
    parser.add_argument("--video-folder", type=str, help=video_folder_help, default="data")
    parser.add_argument("--use-cache", type=str, default=True, help=use_cache_help)
    args = parser.parse_args()
    return args


def main():
    args = get_argument_parser()
    if args.config_file is None:
        print("Input config file path")
    elif args.video_folder is None:
        print("Input path to folder with videos")
    else:
        from eyepreprocessor.patches_preprocessor import EyePreprocessor
        ep = EyePreprocessor(args.config_file)
        ep.get_and_save_patches_from_all_videos_in_folder(args.video_folder, use_cache=args.use_cache)


if __name__ == "__main__":
    main()
