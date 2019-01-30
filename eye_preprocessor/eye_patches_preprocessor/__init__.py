import dlib
import cv2 as cv
from time import time
from eye_preprocessor.utils import *
from imutils import face_utils
import os


class EyePreprocessor:
    """

    EyePreprocessor objects contains face detector and landmark predictor, which intended for eye recognizing and
    cropping recognized areas. Class designed for work with video files, frame-by-frame processing included.

    Attributes
    ----------
    eye_width : int
        Pixel width of resulted eye patch
    eye_height : int
        Pixel height of resulted eye patch
    detector : dlib.fhog_object_detector
        Default dlib face detector
    predictor : dlib.shape_predictor
        Landmark predictor intended for eye recognizing.

    """

    def __init__(self, config_filename=None):
        """

        Initializes preprocessor parameters respectively to configuration file

        Parameters
        ----------
        config_filename : str or None
            Path to file which contains configuration for eye preprocessor. If None, set default values to preprocessor
            params.
        """
        if config_filename is not None:
            conf = get_config_from_file(config_filename)
            print(conf)
            self.eye_width = conf['eye_width']
            self.eye_height = conf['eye_height']
            predictor_path = conf['path_to_landmark_predictor']
        else:
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            self.eye_width = 60
            self.eye_height = 36
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        print(type(self.detector))
        print(type(self.predictor))

    def get_patches_from_video(self, video_filename, use_cache=False, cache_dir="cache"):
        """

        Generator, which yields left eye and right eye patches of first recognized face, frame-by-frame. Respectively
        to given parameters, can be used face bounding boxes cache, which stores in cache_dir. If no faces recognized,
        won't yields, and next frame will be proceeded.

        Parameters
        ----------
        video_filename : str
            Path to video file, which will be proceeded frame-by-frame
        use_cache : bool
            If True, will be used available cache for given video file
        cache_dir : str
            Path to directory which contains cache JSON document with list of saved caches and those saved caches.

        Yields
        -------
        patches : list of ndarray
            len(patches) == 2. patches[0] represents left eye patch, patches[1] represents right eye patch.

        """
        if use_cache:
            caches_filename = os.path.join(cache_dir, "caches.json")
            caches_file = get_caches_json_from_file(caches_filename)
            if video_filename in caches_file:
                cache_filename = os.path.join(cache_dir, caches_file[video_filename])
                faces_cache = load_face_cache(cache_filename)
                number_of_cached_faces = faces_cache.shape[0]
            else:
                cache_filename = str(time()) + '.npy'

                add_new_cache_filename_to_caches_json(caches_filename, caches_file, video_filename, cache_filename)
                cache_filename = os.path.join(cache_dir, cache_filename)
                number_of_cached_faces = 0

        cap = cv.VideoCapture(video_filename)
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if use_cache:
                    if frame_number < number_of_cached_faces:
                        patches = self.get_patches_from_frame(gray, cached_rect=faces_cache[frame_number])
                    else:
                        patches = self.get_patches_from_frame(gray)
                else:
                    patches = self.get_patches_from_frame(gray)
                if patches is not None:
                    if use_cache:
                        if frame_number >= number_of_cached_faces:
                            add_face_to_saved_numpy_array(np.hstack((np.array([1]), np.array(patches[0]))),
                                                          cache_filename)
                    yield patches[1:]
                else:
                    if use_cache:
                        if frame_number >= number_of_cached_faces:
                            add_face_to_saved_numpy_array(np.zeros((1, 5)), cache_filename)
                frame_number += 1
            else:
                break
            # cv.imshow('frame', gray)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def get_patches_from_frame(self, frame, cached_rect=None):
        """

        Returns eye patches and face bounding box of given frame.

        Parameters
        ----------
        frame : ndarray
            Frame with supposed face
        cached_rect : ndarray or None
            If not None, previous cached face bounding box for given frame. cached_rect[0] == 1, means there are faces
            in frame, cached_rect[0] == 0 means there no faces in frame.

        Returns
        -------
        list or None:
            None returns when no faces recognized in frame. Otherwise returns [list of int, ndarray, ndarray], where
            first element represents face bounding box, second - left eye patch, third - right eye patch
        """
        if cached_rect is None:
            rects = self.detector(frame, 1)
            if len(rects) > 0:
                rect = rects[0]
            else:
                return None
        else:
            if cached_rect[0] == 0:
                return None
            else:
                rect = list2dlib(cached_rect[1:])
        shape = self.predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye_landmarks = shape[face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"][0]:
                                   face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"][1]]
        right_eye_landmarks = shape[face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"][0]:
                                    face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"][1]]

        left_eye_roi = self.__get_eye_patch(left_eye_landmarks, frame)
        right_eye_roi = self.__get_eye_patch(right_eye_landmarks, frame)

        return [dlib2list(rect), left_eye_roi, right_eye_roi]

    def __get_eye_patch(self, eye_landmarks, frame):
        """

        Crops eye patch from given frame and eye landmarks

        Parameters
        ----------
        eye_landmarks : ndarray
            Array with shape (num_eye_landmarks, 2) with eye landmark positions.
        frame : ndarray
            Frame with recognized eye

        Returns
        -------
        eye_roi : ndarray
            Resulted eye patch
        """
        (x, y, w, h) = cv.boundingRect(eye_landmarks)
        eye_roi = frame[y:y + h, x:x + w]
        eye_roi = cv.resize(eye_roi, (self.eye_width, self.eye_height))
        eye_roi = cv.equalizeHist(eye_roi)
        return eye_roi


if __name__ == "__main__":
    ep = EyePreprocessor("example_conf.json")
    # for patch in ep.get_patches_from_video("../some_videos/2.avi", cache_dir="../cache", use_cache=False):
    #     name = "../some_videos/"+str(time())+".png"
    #     cv.imwrite(name, patch[0])
    #     print(name, "saved")
