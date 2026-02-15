# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from Tools.config import config


def accuracy(preds, target):
    preds = torch.max(preds, 1)[1].float()
    acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())

    return acc


def compute_mAP(preds, targets):
    preds = torch.softmax(preds, dim=1).detach().cpu().numpy()  # [N, C]
    targets = targets.detach().cpu().numpy()  # [N]
    num_classes = preds.shape[1]
    AP = []

    for i in range(num_classes):
        y_true = (targets == i).astype(int)
        y_score = preds[:, i]

        if y_true.sum() == 0:
            continue

        ap = average_precision_score(y_true, y_score)
        AP.append(ap)

    return sum(AP) / len(AP)

#
# def compute_mAP(preds, target):
#     preds = preds.cpu().numpy()
#     target = target.cpu().numpy()
#     AP = []
#     for i in range(config['num_class']):
#         tgt = target == [i]
#         pred = preds[:,i]
#         AP.append(average_precision_score(tgt, pred))
#     return np.nansum(AP)/config['num_class']


class EvalMetric(object):

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.reset()

    def update(self, preds, labels, losses):
        raise NotImplementedError()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self, prefix=''):
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        name = [prefix + x for x in name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def check_label_shapes(self, preds, labels):
        # raise if the shape is inconsistent
        if (type(labels) is list) and (type(preds) is list):
            label_shape, pred_shape = len(labels), len(preds)
        else:
            label_shape, pred_shape = labels.shape[0], preds.shape[0]

        if label_shape != pred_shape:
            raise NotImplementedError("")


class MetricList(EvalMetric):
    """Handle multiple evaluation metric
    """

    def __init__(self, *args, name="metric_list"):
        assert all([issubclass(type(x), EvalMetric) for x in args]), \
            "MetricList input is illegal: {}".format(args)
        self.metrics = [metric for metric in args]
        super(MetricList, self).__init__(name=name)

    def update(self, preds, labels, losses=None):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels
        losses = [losses] if type(losses) is not list else losses

        for metric in self.metrics:
            metric.update(preds, labels, losses)

    def reset(self):
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                metric.reset()
        else:
            logging.warning("No metric defined.")

    def get(self):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get())
        return ouputs

    def get_name_value(self, **kwargs):
        ouputs = []
        for metric in self.metrics:
            ouputs.append(metric.get_name_value(**kwargs))
        return ouputs


####################
# COMMON METRICS
####################

class Accuracy(EvalMetric):
    """Computes accuracy classification score.
    """

    def __init__(self, name='accuracy', topk=1):
        super(Accuracy, self).__init__(name)
        self.topk = topk

    def update(self, preds, labels, losses=None):
        preds = [preds] if type(preds) is not list else preds
        labels = [labels] if type(labels) is not list else labels

        self.check_label_shapes(preds, labels)
        for pred, label in zip(preds, labels):
            assert self.topk <= pred.shape[1], \
                "topk({}) should no larger than the pred dim({})".format(self.topk, pred.shape[1])
            _, pred_topk = pred.topk(self.topk, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(label.view(1, -1).expand_as(pred_topk))

            self.sum_metric += float(correct.view(-1).float().sum(0, keepdim=True).numpy())
            self.num_inst += label.shape[0]


class Loss(EvalMetric):
    """Dummy metric for directly printing loss.
    """

    def __init__(self, name='loss'):
        super(Loss, self).__init__(name)

    def update(self, preds, labels, losses):
        assert losses is not None, "Loss undefined."
        for loss in losses:
            self.sum_metric += float(loss.numpy().sum())
            self.num_inst += loss.numpy().size


if __name__ == "__main__":
    import torch

    # Test Accuracy
    predicts = [torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]]))]
    labels = [torch.from_numpy(np.array([0, 1, 1]))]
    losses = [torch.from_numpy(np.array([0.3, 0.4, 0.5]))]

    print(torch.from_numpy(np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])).size())
    print(torch.from_numpy(np.array([0, 1, 1])).size())
    print(torch.from_numpy(np.array([0.3, 0.4, 0.5])).size())

    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("input pred:  {}".format(predicts))
    logging.debug("input label: {}".format(labels))
    logging.debug("input loss: {}".format(labels))

    acc = Accuracy()

    acc.update(preds=predicts, labels=labels)

    logging.info(acc.get())

    # Test MetricList
    metrics = MetricList(Loss(name="ce-loss"),
                         Accuracy(topk=1, name="acc-top1"),
                         Accuracy(topk=2, name="acc-top2"),
                         )
    metrics.update(preds=predicts, labels=labels, losses=losses)

    logging.info("------------")
    logging.info(metrics.get_name_value(prefix='ts-'))
    logging.info("------ with prefix ------")
    logging.info(acc.get_name_value(prefix='ts-'))
