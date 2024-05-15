import cv2
import argparse
import numpy as np
from core.facedetector import FaceDetector, ImageFaceExtractor
from utils.associate_detection_trackers import associate_detections_to_trackers
from filterpy.kalman import KalmanFilter
from pathlib import Path
import os
#from deepface import DeepFace

class KalmanTracker(object):
    counter = 1

    def __init__(self, dets):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = np.array([dets[0], dets[1], dets[2], dets[3]]).reshape((4, 1))
        self.id = KalmanTracker.counter
        KalmanTracker.counter += 1

    def __call__(self):
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.kf.x

    def correction(self, measurement):
        self.kf.update(measurement)

    def get_current_x(self):
        bbox = (np.array([self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]]).reshape((1, 4)))
        return bbox

    @classmethod
    def reset_counter(cls):
        cls.counter = 1

class FaceTracker(object):
    def __init__(self):
        self.current_trackers = []

    def __call__(self, detections):
        retain_trackers = []
        if len(self.current_trackers) == 0:
            self.current_trackers = []
            for d in range(len(detections)):
                tracker = KalmanTracker(detections[d, :])
                measurement = np.array((4, 1), np.float32)
                measurement = np.array([[int(detections[d, 0])], [int(detections[d, 1])], [int(detections[d, 2])],
                                        [int(detections[d, 3])]], np.float32)
                tracker.correction(measurement)
                self.current_trackers.append(tracker)
            for trk in self.current_trackers:
                d = trk.get_current_x()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape(1, -1))
            if len(retain_trackers) > 0:
                return np.concatenate(retain_trackers)
            return np.empty((0, 5))
        else:
            predicted_trackers = []
            for t in range(len(self.current_trackers)):
                predictions = self.current_trackers[t]()[:4]
                predicted_trackers.append(predictions)
            predicted_trackers = np.asarray(predicted_trackers)
            matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections[:, :],predicted_trackers)
            #print('Matched Detections & Trackers', len(matched))
            #print('Unmatched Detections', len(unmatched_detections))
            #print('Unmatched Trackers', len(unmatched_trackers))
            #print('Current Trackers', len(self.current_trackers))
            for t in range(len(self.current_trackers)):
                if t not in unmatched_trackers:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    self.current_trackers[t].correction(np.array([detections[d, 0], detections[d, 1],detections[d, 2], detections[d, 3]]).reshape((4, 1)))
            for i in unmatched_detections:
                tracker = KalmanTracker(detections[i, :])
                measurement = np.array((4, 1), np.float32)
                measurement = np.array([[int(detections[i, 0])], [int(detections[i, 1])], [int(detections[i, 2])],[int(detections[i, 3])]], np.float32)
                tracker.correction(measurement)
                self.current_trackers.append(tracker)
            for index in sorted(unmatched_trackers, reverse=True):
                del self.current_trackers[index]
            for trk in self.current_trackers:
                d = trk.get_current_x()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape(1, -1))
        if len(retain_trackers) > 0:
            return np.concatenate(retain_trackers)
        return np.empty((0, 5))

def read_detect_track_faces(videopath, facedetector, display=True):
    facetracker = FaceTracker()
    detection_frame_rate = 5
    videocapture = cv2.VideoCapture(videopath)
    success, frame = videocapture.read()
    frame_number = 1
    faces_per_person = [[np.zeros((224, 224))]]
    original_faces = [-1]
    while success:
        success, frame = videocapture.read()
        if success == False:
            break
        if (frame_number % detection_frame_rate == 0) or (frame_number == 1):
            faces = facedetector.detect(frame)
        if faces.shape[0] == 0:
            continue
        trackers = facetracker(faces)
        print(frame_number, success, len(trackers))
        frame_number += 1
        img = frame.copy()
        ord_image = frame.copy()
        trackers = ImageFaceExtractor.add_margin_to_detection(trackers,ord_image.shape)
        for tracker in trackers:
            tracker = tracker.astype(np.int32)
            person_id = int(tracker[-1])
            face = cv2.resize(ord_image[tracker[1]:tracker[3], tracker[0]:tracker[2], :], (224, 224))
            if len(faces_per_person) <= person_id:
                faces_per_person.append([])
                original_faces.append(person_id)
                # for i in range(1, person_id - 1):
                #     if original_faces[i] != i:
                #         continue
                #     faceresult = DeepFace.verify(faces_per_person[original_faces[i]][0], face,enforce_detection=False)
                #     if faceresult['verified'] == True and faceresult['distance']<=0.15:
                #         print('Distance',faceresult['distance'])
                #         original_faces[person_id] = original_faces[i]
                #         break
                # else:
                #     original_faces[person_id] = person_id
            faces_per_person[original_faces[person_id]].append(face)
            data = {'frame number': str(frame_number + 1), 'person number': str(int(tracker[-1])),
                    'bounding box': str(tracker[:-1])}
            print(data)
            if display:
                cv2.rectangle(img, (tracker[0], tracker[1]), (tracker[2], tracker[3]), (0, 0, 255), 2)
        if display:
            cv2.imshow('Face Tracker', cv2.resize(img, (320, 240)))
            if cv2.waitKey(1) == ord('q'):
                return
    return faces_per_person

def parse_args():
    parser = argparse.ArgumentParser(description='Tracking Arguments')
    parser.add_argument('--videofile', help='Input Video File',required=False)
    parser.add_argument('--videopath',help='Input Video Path',required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    detector_name = "mymodel"
    video_file_path = args.videofile
    video_dir_path = args.videopath
    if video_dir_path is None:
        print('Single File Mode')
        output_path = f'./test/output'
        facedetector = FaceDetector(detector_name)
        result = read_detect_track_faces(video_file_path, facedetector, True)
        os.makedirs(output_path,exist_ok=True)
        basefilename = os.path.basename(video_file_path).replace('.', '_')
        for i, frames in enumerate(result):
            if i == 0:
                continue
            if len(frames) == 0:
                continue
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{output_path}/{basefilename}_person_{i}.mp4', fourcc, 30.0, (224, 224))
            for frame in frames:
                out.write(frame)
            out.release()
    else:
        print('Video Directory Mode')
        video_files = list(Path(video_dir_path).rglob("*.mp4"))
        for videofilename in video_files:
            if videofilename.is_file():
                relative_path = videofilename.relative_to(video_dir_path)
                #print(relative_path)
                KalmanTracker.reset_counter()
                videofilename = str(videofilename)
                #print(videofilename)
                output_path = Path('test') / 'output' / relative_path
                #print(output_path)
                facedetector = FaceDetector(detector_name)
                result = read_detect_track_faces(videofilename, facedetector, True)
                os.makedirs(output_path, exist_ok=True)
                basefilename = os.path.basename(videofilename).replace('.', '_')
                for i, frames in enumerate(result):
                    if i == 0:
                        continue
                    if len(frames) == 0:
                        continue
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(f'{output_path}/{basefilename}_person_{i}.mp4', fourcc, 30.0, (224, 224))
                    for frame in frames:
                        out.write(frame)
                    out.release()