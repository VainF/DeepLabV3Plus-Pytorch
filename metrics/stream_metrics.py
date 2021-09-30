import typing as t
import warnings

import numpy as np
import torch
from torch import Tensor, isnan


def torchnanmean(v: torch.Tensor, *args, allnan=np.nan, **kwargs) -> torch.Tensor:
    """

    :param v: tensor to take mean
    :param dim: dimension(s) over which to take the mean
    :param allnan: value to use in case all values averaged are NaN.
        Defaults to np.nan, consistent with np.nanmean.
    :return: mean.
    """
    v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0

    if np.isnan(allnan):
        return v.sum(*args, **kwargs) / float(~is_nan).sum(*args, **kwargs)
    else:
        sum_nonnan = v.sum(*args, **kwargs)
        n_nonnan = float(~is_nan).sum(*args, **kwargs)
        mean_nonnan = torch.zeros_like(sum_nonnan) + allnan
        any_nonnan = n_nonnan > 1
        mean_nonnan[any_nonnan] = (
            sum_nonnan[any_nonnan] / n_nonnan[any_nonnan])
        return mean_nonnan


class _StreamMetrics(t.Protocol):

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, device="cpu"):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self) -> t.Dict[str, t.Union[t.Dict[int, float], t.Dict[str, float]]]:
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        with warnings.catch_warnings:
            warnings.simplefilter("ignore")

            hist = self.confusion_matrix
            acc = np.diag(hist).sum() / hist.sum()
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            mean_iu = np.nanmean(iu)
            freq = hist.sum(axis=1) / hist.sum()
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
            cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class StreamSegMetricsCUDA(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, device="cpu"):
        self.n_classes = n_classes
        self._device = device
        self.reset()

    def update(self, label_trues: Tensor, label_preds: Tensor):
        label_trues, label_preds = label_trues.detach(), label_preds.detach()
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = torch.bincount(
            self.n_classes * label_true[mask].long() + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self) -> t.Dict[str, t.Union[t.Dict[int, float], t.Dict[str, float]]]:
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / hist.sum(dim=1)
        acc_cls = torchnanmean(acc_cls)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iu = torchnanmean(iu)
        freq = hist.sum(dim=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), device=self._device)


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
