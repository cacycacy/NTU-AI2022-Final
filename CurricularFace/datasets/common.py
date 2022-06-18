import copy
import torch
import operator


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def compute_cosine_similarity(enrollment_embeddings, labels, init_ids_times,faces, objs, model, k=10):
    with torch.no_grad():
        face_embeddings = model(faces)

    face_embeddings = l2_norm(face_embeddings)
    enrollment_embeddings = l2_norm(enrollment_embeddings).T
    cosine_similarity = torch.mm(face_embeddings, enrollment_embeddings)
    scores, indices = cosine_similarity.cpu().topk(k, dim=1)
    scores, indices = scores.numpy(), indices.numpy()
    for i, obj in enumerate(objs):
        ids_times = copy.deepcopy(init_ids_times)
        for score, index in zip(scores[i], indices[i]):
            ids_times[labels[index]] += 1
            obj.dict_ids[labels[index]] += score

        for ID, times in ids_times.items():
            if times > 1:
                obj.dict_ids[ID] /= times
            obj.dict_ids[ID] *= (times / k) ** 0.5      # Predict distribution

        ID, times = max(ids_times.items(), key=operator.itemgetter(1))
        score = obj.dict_ids[ID]

        # If without obj.ID, obj.ID_score, the sort will not update average_ids_dict.
        if times > (k * 0.5) and score >= 0.35:
            obj.ID  = ID
            obj.ID_score = obj.dict_ids[ID]

    return objs