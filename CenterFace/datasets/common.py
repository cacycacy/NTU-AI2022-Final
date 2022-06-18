import os
import cv2
import copy
import random
import operator
import numpy as np
import math
from Config import Config as conf

class BBox:
    def __init__(self, cls, xyrb, dict_ids, score=0, landmark=None, rotate=False):
        self.cls = cls              # Class 0: non-occlusion / 1: occlusion.
        self.score = score          # The obj is face confidence score.
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate        # Training needs.
        self.reid = None            # Demo tracking.
        self.num_matching = 0       # Record the number of matches.
        self.dict_ids = dict_ids    # Record the ID & ID score.
        self.ID = "Unknown"         # According to KNN search, who(ID) is this obj.
        self.ID_score = None        # According to KNN search, the score of the corresponding ID.
        self.dict_ids_times = None  # Dictionary ID Times
        self.First_face = None      # GUI display
        self.start_time = None
        self.end_time = None
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join(
            [str(item[:2]) for item in self.landmark]) if self.landmark is not None else "empty"
        return f"(BBox[{self.cls}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
               f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def classification(self):
        return self.cls

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    # [[x, y], [x, y], [x, y], [x, y], [x, y]] to [x, x, x, x, x, y, y, y, y, y]
    @property
    def x5y5_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def id(self):
        return self.reid

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    @property
    def convert_bbox_to_z(self):
        return np.array([(self.x + self.r) * 0.5, (self.y + self.b) * 0.5,
                         self.area, self.width/float(self.height)]).reshape((4, 1))


    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou

def determine_cls(x, y, r, b, w, h, cls):
    """
        Process bounding box exceeds the image size. default [800, 800]
        Determine exceed ratio: 0.25
        if true:    occlusion --> cls=1
        else:    keep the original cls
    """
    max_w, max_h = conf.insize
    ratio = 0.25
    min_x = -ratio * w
    min_y = -ratio * h
    max_x = max_w + ratio * w
    max_y = max_h + ratio * h
    if x < min_x or y < min_y or r > max_x or b > max_y:
        cls = 1
    return cls

def intv(*value):
    if len(value) == 1:
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def floatv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([float(item) for item in value])
    elif isinstance(value, list):
        return [float(item) for item in value]
    elif value is None:
        return 0
    else:
        return float(value)


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low


def drawbbox(frame, obj, non_occ_color=(0, 0, 255), occ_color=(0, 255, 0), unknown_color=(150, 150, 150), \
             thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255)):

    # text = f"{obj.score:.2f}"
    x, y, r, b = intv(obj.box)
    w = r - x + 1
    h = b - y + 1
    border = int(thickness / 2)
    pos = (x + 3, y - 5)

    if obj.classification == 0:
        bbox_color = non_occ_color
    else:
        bbox_color = occ_color
    border_color = bbox_color
    if obj.ID == 'Unknown':
        border_color = unknown_color
    # 畫信心%數
    # if obj.ID_score != None:
    #     score_h = int(h * obj.ID_score)
    #     score_w = max(int(w * 0.1), 1)
    #     cv2.rectangle(frame, (r + border, b-score_h, score_w, score_h + thickness), bbox_color, -1)
    cv2.rectangle(frame, (x, y, w, h), bbox_color, thickness)
    cv2.rectangle(frame, (x - border, y - 21, w + thickness, 21), border_color, -1)
    cv2.putText(frame, obj.ID, pos, 0, 0.5, textcolor, 1)
    # cv2.putText(frame, text, pos, 0, 0.5, textcolor, 1)


    if obj.haslandmark:
        for i in range(len(obj.landmark)):
            x, y = obj.landmark[i][:2]
            cv2.circle(frame, intv(x, y), 3, landmarkcolor, -1, 16)


def gaussian_truncate_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h_radius = math.ceil(h_radius)
    w_radius = math.ceil(w_radius)
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_truncate_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return gaussian


def truncate_radius(sizes, stride=4):
    return [s / (stride * 4.0) for s in sizes]


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = math.ceil(det_size[0]), math.ceil(det_size[1])

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)




def label2index_to_index2label(label2index_dic):
    index2label = {}
    for label in label2index_dic:
        index = label2index_dic[label]
        index2label[index] = label
    return index2label


def parse_facials_webface(facials):
    ts = []
    for facial in facials:
        x, y, w, h = facial[:4]
        box = [x, y, x + w - 1, y + h - 1]
        landmarks = None

        if w * h < 4 * 4:
            continue

        if len(facial) >= 19:
            landmarks = []
            for i in range(5):
                x, y, t = facial[i * 3 + 4:i * 3 + 4 + 3]
                if t == -1:
                    landmarks = None
                    break

                landmarks.append([x, y])

        ts.append(BBox(cls=0, xyrb=box, landmark=landmarks, rotate=False))
    return ts


def load_webface(labelfile, imagesdir):
    with open(labelfile, "r") as f:
        lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]

    facials = []
    file = None
    files = []
    for index, line in enumerate(lines):
        if line.startswith("#"):
            if file is not None:
                files.append([f"{imagesdir}/{file}", parse_facials_webface(facials)])

            file = line[2:]
            facials = []
        else:
            facials.append([float(item) for item in line.split(" ")])

    if file is not None:
        files.append([f"{imagesdir}/{file}", parse_facials_webface(facials)])
    return files


def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image


def log(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [log(item) for item in v]

    elif isinstance(v, np.ndarray):
        return np.array([log(item) for item in v])

    base = np.exp(1)
    if abs(v) < base:
        return v / base

    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]

    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v])

    gate = 1
    if abs(v) < gate:
        return v * np.exp(gate)

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def crop_face(image, obj):

    # Alignment: using eye center point & mouth center point
    left_eye = np.array(obj.landmark[0])
    right_eye = np.array(obj.landmark[1])
    center_eye = (left_eye + right_eye) / 2
    left_mouth = np.array(obj.landmark[3])
    right_mouth = np.array(obj.landmark[4])
    center_mouth = (left_mouth + right_mouth) / 2

    vector = center_eye - center_mouth

    angle = 90 + np.angle(vector[0] + vector[1] * 1j) / np.pi * 180

    M = cv2.getRotationMatrix2D(center=tuple(obj.center), angle=angle, scale=1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    x, y, r, b = intv(obj.box)

    w = int(obj.width)
    h = int(obj.height)

    if w > h:
        x_ = x
        r_ = r
        offset = (w - h) // 2
        y_ = y - offset
        b_ = b + offset
    else:
        y_ = y
        b_ = b
        offset = (h - w) // 2
        x_ = x - offset
        r_ = r + offset

    face = np.zeros((b_ - y_ + 1, r_ - x_ + 1, 3), dtype=image.dtype)

    off_x, off_y, off_r, off_b = 0, 0, 0, 0
    if x_ < 0:
        off_x = 0 - x_
        x_ = max(x_, 0)
    if y_ < 0:
        off_y = 0 - y_
        y_ = max(y_, 0)
    if r_ >= image.shape[1]:
        off_r = r_ - image.shape[1] + 1
        r_ = min(r_, image.shape[1] - 1)
    if b_ >= image.shape[0]:
        off_b = b_ - image.shape[0] + 1
        b_ = min(b_, image.shape[0] - 1)

    face[off_y:-off_b-1, off_x:-off_r-1, :] = image[y_:b_, x_:r_, :]
    face = cv2.resize(face, (112, 112))
    obj.First_face = face
    return face


class AverageMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

class Record_ID(object):
    """
        Record the id identified by obj in the past
        input: obj.dict_ids
    """
    def __init__(self, init_dict_ids):
        self.past_dict_ids = []
        self.count = 1
        self.reset_id(init_dict_ids)    # Initial dict_ids
    def reset_id(self, init_dict_ids):
        self.dict_ids = init_dict_ids
        self.avg_dict_ids = copy.deepcopy(self.dict_ids)
        self.dict_ids_times = {key:0 for key in init_dict_ids.keys()}

    def update(self, new_dict_ids, predict_id, id_score):
        if id_score != None:
            self.count += 1
            self.dict_ids_times[predict_id] += 1
            self.past_dict_ids.append(new_dict_ids)
            for key, value in new_dict_ids.items():
                self.dict_ids[key] += value

            if len(self.past_dict_ids) > 30:
                remove_dict_ids = self.past_dict_ids.pop(0)
                for key, value in remove_dict_ids.items():
                    self.dict_ids[key] -= value
                self.count -= 1

            self.avg_dict_ids = {key:(value/self.count) for key, value in self.dict_ids.items()}

        # print(self.avg_dict_ids)