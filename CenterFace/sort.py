"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter

from CenterFace.datasets.common import computeIOU, Record_ID


def convert_x_to_bbox(x, score=None):
    """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if score == None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
        This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, obj):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity models
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.   # default: 10
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.3     # default: 0.01
        self.kf.Q[4:,4:] *= 0.3     # default: 0.01

        self.kf.x[:4] = obj.convert_bbox_to_z
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.dict_ids = Record_ID(obj.dict_ids)
        self.start_time = obj.start_time
        self.end_time = None
    def update(self, obj):
        """
            Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(obj.convert_bbox_to_z)
        self.dict_ids.update(obj.dict_ids, obj.ID, obj.ID_score)

    def predict(self):
        """
            Advances the state vector and returns the predicted bounding box estimate.
        """
        if(self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
            Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.5):
    """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, obj in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = computeIOU(obj.box, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, obj in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []

    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches)==0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1):
        """
            Sets key parameters for SORT
        """
        self.max_age = max_age
        self.trackers = []
        self.exist_reid = []

        self.frame_count = 0
    def update(self, objs):
        """
            Params:
              dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections.
            Returns the a similar array, where the last column is the object ID.

            NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(objs, trks)

        # only match id
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(objs[d][0])
                objs[d][0].reid = trk.id
                # objs[d][0].box = trk.get_state()[0].tolist()
                trk.end_time = objs[d][0].start_time
                objs[d][0].start_time = trk.start_time
                objs[d][0].end_time = trk.end_time
                objs[d][0].num_matching = trk.hits
                objs[d][0].dict_ids = trk.dict_ids.avg_dict_ids
                objs[d][0].dict_ids_times = trk.dict_ids.dict_ids_times
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(objs[i])
            self.exist_reid.append(trk.id)
            objs[i].reid = trk.id
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # remove dead tracklet
            i -= 1
            if trk.time_since_update > self.max_age:
                trk_remove = self.trackers.pop(i)
                self.exist_reid.remove(trk_remove.id)
        return sorted(objs, key=lambda obj:obj.reid, reverse=False), set(self.exist_reid)