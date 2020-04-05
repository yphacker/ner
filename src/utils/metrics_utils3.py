import torch
from collections import Counter
from conf import config_span as config


class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = config.id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def get_score(true_subjects, pred_subjects):
    # print(true_subjects, pred_subjects)
    metric = SpanEntityScore(config.id2label)
    for i in range(len(true_subjects)):
        metric.update(true_subjects[i], pred_subjects[i])

    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    # print("***** results *****")
    # info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    # print(info)
    # print("***** entity results %s *****")
    # for key in sorted(entity_info.keys()):
    #     print("******* %s results ********" % key)
    #     info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
    #     print(info)
    return results['acc']
