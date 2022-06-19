import os
import operator
import collections

import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# detection
from Config import Config as conf
from CenterFace.models.centerface import CenterFace
from CenterFace.demo import detect
from CenterFace.datasets import common
from CenterFace.sort import Sort
# recognition
from CurricularFace.models.seesawfacenet import SeesawFaceNet
from CurricularFace.datasets.enrollment_dataset import Load_Enroll_Dataset
from CurricularFace.datasets.common import compute_cosine_similarity



class FaceDetectorWithID():
    def __init__(self):
        self.device = conf.device
        self.load_models()
        self.load_enrollments()


    def load_models(self):
        self.detect_model = CenterFace().eval().to(self.device)
        self.detect_model.load_state_dict(torch.load(f"./CenterFace/checkpoints/{conf.load_detect_model}"))
        self.recog_model = SeesawFaceNet().eval().to(self.device)
        self.recog_model.load_state_dict(torch.load(f"./CurricularFace/checkpoints/{conf.load_recog_model}"))


    def load_enrollments(self, batch_size=25):
        print(f"enrolled members: {len(os.listdir(conf.member_path))}")
        enrollment_dict = collections.OrderedDict()
        enrollment_dataset = Load_Enroll_Dataset(conf.member_path)
        enrollment_loader = DataLoader(dataset=enrollment_dataset, batch_size=batch_size, shuffle=False)
        for i, (img_paths, imgs) in enumerate(tqdm(enrollment_loader, desc=f"load ID", total=len(enrollment_loader))):
            with torch.no_grad():
                imgs = imgs.to(self.device)
                embeddings = self.recog_model(imgs)
            res = {img_path: embedding[None, :] for (img_path, embedding) in zip(img_paths, embeddings)}
            enrollment_dict.update(res)
        self.labels = [keys.split('/')[2] for keys in enrollment_dict.keys()]
        self.enrollment_embeddings = torch.cat(list(enrollment_dict.values())).to(self.device)
        self.initial_dict_ids = {id: 0 for id in set(self.labels)}
        # print(self.initial_dict_ids)


    def start(self):
        self.mot_tracker = Sort(30)
        self.cap = cv2.VideoCapture(conf.cap_index)
        while(True):
            self.ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if self.ret:
                objs = detect(self.detect_model, frame, self.initial_dict_ids, frame.shape[0] * frame.shape[1],
                                    threshold=conf.threshold, device=self.device)
                if len(objs) > 0:
                    faces = []
                    for obj in objs:
                        faces.append(conf.recog_transforms(common.crop_face(frame, obj)))
                    faces = torch.stack(faces, dim=0).to(self.device)
                    objs, _ = self.mot_tracker.update(compute_cosine_similarity(self.enrollment_embeddings, self.labels, self.initial_dict_ids,
                                                        faces, objs, self.recog_model, k=conf.knn_num))           

                    for obj in objs:
                        obj.ID = max(obj.dict_ids.items(), key=operator.itemgetter(1))[0]
                        obj.ID_score = obj.dict_ids[obj.ID]
                        # For Unknown ID
                        if obj.num_matching > 3:
                            if (obj.cls == 0 and obj.ID_score < conf.recog_th) or (obj.dict_ids_times[obj.ID] / obj.num_matching < 0.2):
                                obj.ID, obj.ID_score = "Unknown", None
                            elif (obj.cls == 1 and obj.ID_score < conf.recog_th - 0.05) or (obj.dict_ids_times[obj.ID] / obj.num_matching < 0.2):
                                obj.ID, obj.ID_score = "Unknown", None
                            common.drawbbox(frame, obj)
                    cv2.imshow('Result', frame)

                else:  # without obj
                    objs, _ = self.mot_tracker.update(objs) 
                    cv2.imshow('Result', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break



if __name__ == "__main__":
    OurSystem = FaceDetectorWithID()
    OurSystem.start()