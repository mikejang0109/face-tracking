from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.abs import GlobalInstanceAbstract
from utils.utils import monitor_execution_time


class FaceDetector(object):
    def __init__(self, detectorname):
        if detectorname == 'mymodel':
            self.detector = ImageFaceExtractor()
        pass
    def detect(self, image):
        return self.detector(image)

class ImageFaceExtractor(GlobalInstanceAbstract):
    def __init__(self):
        super().__init__()
        self.__tf_face_detector = cv2.dnn.readNetFromTensorflow('./models/det_uint8.pb', './models/det.pbtxt')

    @staticmethod
    def predict_bbox(detections, width, height):
        result_list = []
        for i in range(detections.shape[2]):
            c = detections[0, 0, i, 2]
            if c < 0.9:
                continue
            xx1 = int(detections[0, 0, i, 3] * width)
            yy1 = int(detections[0, 0, i, 4] * height)
            xx2 = int(detections[0, 0, i, 5] * width)
            yy2 = int(detections[0, 0, i, 6] * height)
            if yy1 > height or yy2 > height or xx1 > width or xx2 > width:
                continue
            result_list.append([xx1, yy1, xx2, yy2])
        return np.array(result_list)

    def __call__(self, org_image: str | Path | np.ndarray, threshold: float = 0.9):
        if not isinstance(org_image, np.ndarray):
            org_image = cv2.imread(org_image)

        tf_blob = cv2.dnn.blobFromImage(org_image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.__tf_face_detector.setInput(tf_blob)
        detections = self.__tf_face_detector.forward()
        boxes = self.predict_bbox(detections, org_image.shape[1], org_image.shape[0])
        #return self._crop_face_from_bbox(org_image, boxes)
        #print(boxes)
        return self._add_margin_to_detection(boxes, org_image.shape)

    def _crop_face_from_bbox(self, image: np.ndarray, bboxes: np.ndarray):
        result_bboxes = self._add_margin_to_detection(bboxes, image.shape)
        result_faces = []
        for i in range(result_bboxes.shape[0]):
            bbox = result_bboxes[i]
            ymin, xmin, ymax, xmax = bbox
            face = image[xmin:xmax, ymin:ymax, :]
            result_faces.append(face)
        return np.array(result_faces)

    def _add_margin_to_detection(self, bboxes: np.ndarray, frame_size: Tuple[int, int], margin: float=0.2):
        result_bbox = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            offset = np.round(margin * (bbox[2] - bbox[0]))
            size = int(bbox[2] - bbox[0] + offset * 4)
            center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            half_size = size // 2
            bbox = bbox.copy()
            bbox[0] = max(center[0] - half_size, 0)
            bbox[1] = max(center[1] - half_size, 0)
            bbox[2] = min(center[0] + half_size, frame_size[1])
            bbox[3] = min(center[1] + half_size, frame_size[0])
            result_bbox.append(bbox)
        return np.array(result_bbox)

    def _preprocess(self, org_image: np.ndarray):
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image


class VideoFaceExtractor(GlobalInstanceAbstract):
    NB_FRAMES = 32

    def __init__(self):
        self.__image_face_extractor = ImageFaceExtractor()

    def __call__(self, video_path: str | Path):
        cap = cv2.VideoCapture(str(video_path))
        faces = []
        frames = []
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frame_cnt)
        if frame_cnt <= 0:
            while True:
                flg, frame = cap.read()
                if not flg:
                    break
                frames.append(frame)
            frame_idxs = np.unique(np.linspace(0, len(frames) - 1, self.NB_FRAMES, endpoint=True, dtype=np.int32))
            frames = np.array(frames)[frame_idxs]
        else:
            frame_idxs = np.unique(np.linspace(0, frame_cnt -1 , self.NB_FRAMES, endpoint=True, dtype=np.int32))
            print(frame_idxs)
            for frame_idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        for frame in frames:
            face = self.__image_face_extractor(frame)
            if face is None:
                continue
            faces.append(face)
        return faces


class VideoRealFakeDetector(GlobalInstanceAbstract):
    NB_FRAMES = 32

    def __init__(self):
        super().__init__()
        self.__real_fake_detector = cv2.dnn.readNetFromONNX('./models/dfdc.onnx')
        self.__video_face_extractor = VideoFaceExtractor()

    @property
    def real_fake_detector(self):
        return self.__real_fake_detector

    @monitor_execution_time()
    def __call__(self, video_path: str | Path, boolean: bool = True, boolean_threshold: float = 0.5, ret_faces=False, offset=2.):
        faces = self.__video_face_extractor(video_path)

        faces = np.array([
            self.preprocess_faces(face)
            for face in faces
        ])

        if len(faces) == 0:
            return None

        self.__real_fake_detector.setInput(faces)
        faces_pred = self.__real_fake_detector.forward()

        faces_pred = faces_pred + offset
        faces_pred =  1 / (1 + np.exp(-faces_pred))
        avg_score = faces_pred.mean()

        if ret_faces:
            return avg_score > boolean_threshold if boolean else avg_score, list(zip(faces, faces_pred))
        return avg_score > boolean_threshold if boolean else avg_score

    @staticmethod
    def preprocess_faces(face_img, target_size=(224, 224)):
        face_img = cv2.resize(face_img, target_size).astype(np.float32) / 255
        image_mean = np.array([0.485, 0.456, 0.406])
        image_std = np.array([0.229, 0.224, 0.225])
        face_img = (face_img - image_mean) / image_std
        face_img = np.transpose(face_img, [2, 0, 1])
        face_img = face_img.astype(np.float32)
        return face_img

    def _read_frames_at_indices(self, capture: cv2.VideoCapture, frame_idxs):
        frames = []

        for frame_idx in frame_idxs:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = capture.read()
            if not ret:
                continue

            frames.append(frame)

        return frames

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]

        if self.insets[1] > 0:
            H = frame.shape[1]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]

        return frame


__all__ = [
    'ImageFaceExtractor',
    'VideoRealFakeDetector',
    'FaceDetector',
]
