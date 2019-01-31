import json
import os
import numpy as np
import dlib
from pathlib import Path

def get_config_from_file(filename):
    """

    Returns dict with parameters of eye preprocessor.

    Parameters
    ----------
    filename : str
        Path to file with preprocessor configuration. File must be instance of JSON document.

    Returns
    -------
    dict or None
        Dict with configuration for eye preprocessor. If file with given filename doesn't exist, return None instead.
    """
    try:
        with open(filename, encoding='utf-8') as data_file:
            data = json.loads(data_file.read())
        return data
    except FileNotFoundError:
        print(filename, "doesn't exist")
        return None


def get_caches_json_from_file(filename):
    """

    Returns dict of caches, which saved in file with given filename. If file with this filename doesn't exist, will be
    created new file with given filename with empty dict.

    Parameters
    ----------
    filename : pathlib.Path
        Path to file which contains list of caches. File must be instance of JSON document.

    Returns
    -------
    dict
        Dict, where key is filename of video, value is filename of saved numpy array for this video.
    """
    if filename.exists():
        with filename.open() as data_file:
            data = json.loads(data_file.read())
        return data
    else:
        data = json.loads('{}')
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open(mode='w') as outfile:
            json.dump(data, outfile)
        return data


def add_new_cache_filename_to_caches_json(caches_filename, caches, video_filename, cache_filename):
    """

    Adds to existed dict of caches data about cache of new video file

    Parameters
    ----------
    caches_filename : str
        Path to file, which contains list of caches. File must be instance of JSON document.
    caches : dict
        Existed dict of video filenames and filenames of respective cache files
    video_filename : str
        Filename of new video
    cache_filename
        Filename of cache file of new video
    """
    caches[video_filename] = cache_filename
    with open(caches_filename, 'w') as outfile:
        json.dump(caches, outfile)


def load_face_cache(filename):
    """

    Loads and returns cache, which saved in file with given filename.

    Parameters
    ----------
    filename : pathlib.Path
        Path to file which contains numpy array

    Returns
    -------
    ndarray
        Array with shape (number_of_processed_frames, 5). Each row represents bounding box of previous recognized face
        in particular frame. If first element equals 0, it represents that no face in particular frame.
    """
    return np.load(filename)


def add_face_to_saved_numpy_array(face, array_filename):
    """

    Adds face bounding box of new frame to existed cache. If it's first frame, creates new file with given filename,
    which contains cache.

    Parameters
    ----------
    face : ndarray
        Array with shape (1, 5) which represents bounding box of face of particular frame. If face[0,0] == 0, no face
        recognized in this frame.
    array_filename : str
        Path to file which contains cache of video
    """
    if os.path.exists(array_filename):
        old_arr = np.load(array_filename)
        np.save(array_filename, np.vstack((old_arr, face)))
    else:
        np.save(array_filename, face)


def dlib2list(rect):
    """

    Converts dlib.rectangle object to list

    Parameters
    ----------
    rect : dlib.rectangle
        Represents bounding box of recognized face

    Returns
    -------
    list
        List with rect attributes
    """
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]


def list2dlib(rlist):
    """

    Converts list to dlib.rectagle object

    Parameters
    ----------
    rlist : list
        List which represents bounding box of recognized face

    Returns
    -------
    dlib.rectangle

    """
    return dlib.rectangle(left=rlist[0], top=rlist[1], right=rlist[2], bottom=rlist[3])


def get_all_video_filenames_from_folder(folder, video_formats=['.mp4', '.avi']):
    """

    Search all video files with given formats in given folder

    Parameters
    ----------
    folder : pathlib.Path
        Object that represents folder with videos.
    video_formats : list of str
        List of video formats, which must be proceeded.

    Returns
    -------
    result : list of pathlib.Path
        List of pathes of all video files in given folder
    """
    result = []
    for f in video_formats:
        result.extend(list(folder.glob("**/*"+f)))
    return result


if __name__ == "__main__":
    print(type(get_all_video_filenames_from_folder(Path("../../some_videos"))[0]))
