import numpy as np
import torch
import torch.nn.functional as F
import cv2
import copy
import time
from CenterFace.config import Config as conf
from CenterFace.models.centerface import CenterFace
from CenterFace.datasets import common
from CenterFace.sort import Sort
def nms(objs, iou=0.5):
    """
    Use obj.score to sort the items and compare the iou of each obj from high score to low score.
    The high score obj has priority.

    :param objs: models detect result
    :param iou: iou threshold
    :return: after nms obj
    """
    if objs is None or len(objs) <= 1:
        return np.array(objs)

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)     # if delete flags[i] = 1
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue
        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return np.array(keep)

def detect(model, image, dict_ids, frame_area, threshold=conf.threshold, nms_iou=0.5, device=conf.device):
    image = common.pad(image)      # padding can divide stride=32
    image = conf.transforms(image)[None].to(device)
    with torch.no_grad():
        hm, box, landmark = model(image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)     # nms of centernet (filt some obj with high repeatability)
    scores, indices = ((hm == hm_pool).float() * hm).view(2, -1).cpu().topk(200)

    hm_height, hm_width = hm.shape[2:]

    indices = indices.squeeze()
    ys = (indices // hm_width).numpy()
    xs = (indices  % hm_width).numpy()
    scores = scores.squeeze().numpy()
    box = box.cpu().squeeze().numpy()
    landmark = landmark.cpu().squeeze().numpy()

    stride = 4
    objs = []
    for cls in range(hm.shape[1]):
        for cx, cy, score in zip(xs[cls], ys[cls], scores[cls]):
            if score < threshold:
                break
            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride

            if ((abs(xyrb[2] - xyrb[0]) + 1) * (abs(xyrb[3] - xyrb[1]) + 1)) < (frame_area * 0.005):
                continue

            x5y5 = landmark[:, cy, cx]
            x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(common.BBox(cls, xyrb=xyrb, dict_ids=copy.deepcopy(dict_ids), score=score, landmark=box_landmark))

    return nms(objs, iou=nms_iou)

def image_demo(img_path):
    model = CenterFace()
    model.eval()
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.checkpoints + '/' + conf.load_model))

    image = cv2.imread(img_path)
    objs = detect(model, image)
    for obj in objs:
        common.drawbbox(image, obj)
    cv2.imwrite("test.jpg", image)
    cv2.imshow("image", image)
    cv2.waitKey(0)

def video_demo(vidoe_path):
    model = CenterFace()
    model.eval()
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.checkpoints + '/' + conf.load_model))
    mot_tracker = Sort(5)

    cap = cv2.VideoCapture(vidoe_path)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # save video
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    encoder = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter("Demo_video.mp4", encoder, 30, (w, h), True)

    t_start = time.time()
    while ret:
        t1 = time.time()
        ret, frame = cap.read()
        if ret == False:
            break
        objs = detect(model, frame)
        objs = mot_tracker.update(objs)

        for obj in objs:
            if obj.num_matching >= 3:
                common.drawbbox(frame, obj)
        t2 = time.time()
        fps = round(1 / (t2 - t1), 3)
        cv2.putText(frame, f"FPS:{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        save_video.write(frame)     # save video
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    t_end = time.time()
    print(f"Inference time:{t_end - t_start}, FSP:{num_frame / (t_end - t_start)}")
    cap.release()
    save_video.release()    # save video
    cv2.destroyAllWindows()

def camera_demo():
    model = CenterFace()
    model.eval()
    model.to(conf.device)
    model.load_state_dict(torch.load(conf.checkpoints + '/' + conf.load_model))
    mot_tracker = Sort(3)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # save video
    # h, w = frame.shape[:2]
    # encoder = cv2.VideoWriter_fourcc(*"mp4v")
    # save_video = cv2.VideoWriter("Demo_video.mp4", encoder, 30, (w, h), True)


    while ret:
        t1 = time.time()
        ret, frame = cap.read()
        objs = detect(model, frame)
        # objs has been updated, exist_id is used to vote (recognition)
        objs = mot_tracker.update(objs)

        for obj in objs:
            if obj.num_matching >= 3:
                common.drawbbox(frame, obj)
        t2 = time.time()
        fps = round(1 / (t2 - t1), 3)
        cv2.putText(frame, f"FPS:{fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        # save_video.write(frame)     # save video
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    # save_video.release()    # save video
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # image_demo("../test.jpg")
#     # video_demo("../mv_1.mp4")
#     # camera_demo()

