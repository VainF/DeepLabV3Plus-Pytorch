import numpy as np
from .ms_ssim import MultiScaleSSIM
from skimage.measure import compare_psnr
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

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
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
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



class StreamCompMetrics(_StreamMetrics):
    """
    Stream Metrics for Image Compression Task
    """
    def __init__(self):
        self._psnr = []
        self._ms_ssim = []

    def update(self, ori_img, pred_img):
        assert len(ori_img.shape)==4, ori_img.shape
        for i in range(len(ori_img)):
            self._psnr.append( compare_psnr(ori_img[i], pred_img[i]) )
        self._ms_ssim.append( MultiScaleSSIM(ori_img, pred_img) )

    def get_results(self):
        return (
            np.mean(self._psnr),
            np.mean(self._ms_ssim)
        )
    @staticmethod
    def to_str(result):
        string = "Psnr: %f\nMS-SSIM: %f"%(result[0], result[1])
        return string

    def reset(self):
        del self._psnr
        del self._ms_ssim
        self._psnr = []
        self._ms_ssim = []

class StreamClsMetrics(_StreamMetrics):
    """
    Stream Metrics for Classification Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix[lp][lt] += 1

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %f\n"%(k, v)
        return string
    
    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
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
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]

class VocClsMetrics(_StreamMetrics):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros(shape=(n_classes, 4))  # n_classes, TN, FP, FN, TP  
                                                                            #00, 01, 10, 11 target-predict

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            idx = (tuple(range(self.n_classes)), lt*2+lp)
            self.confusion_matrix[idx] += 1
    @staticmethod
    def to_str(results):
        string = "\n"
        string += "Overall Acc: %f\n"%(results['Overall Acc'])
        string += "Overall Precision: %f\n"%(results['Overall Precision'])
        string += "Overall Recall: %f\n"%(results['Overall Recall'])
        string += "Overall F1: %f\n"%(results['Overall F1'])

        string+='Class Metrics:\n'
        
        for i in range(self.n_classes):
            string += "\tclass %d: acc=%f, precision=%f, recall=%f, f1=%f\n"%(i,results['Class Acc'][i],results['Class Precision'][i],results['Class Recall'][i],results['Class F1'][i]  )
        return string

    def get_results(self):
        TN = self.confusion_matrix[:, 0]
        FP = self.confusion_matrix[:, 1]
        FN = self.confusion_matrix[:, 2]
        TP = self.confusion_matrix[:, 3]

        class_accuracy  = np.nan_to_num( ( TN+TP ) / (TN+FP+FN+TP) )
        class_precision = np.nan_to_num( TP / ( TP+FP ) )
        class_recall    = np.nan_to_num( TP / ( TP+FN ) )
        class_f1        = np.nan_to_num( 2* (class_precision * class_recall) / (class_precision+class_recall) )

        return {'Overall Acc': class_accuracy.mean(), 
                'Overall Precision': class_precision.mean(), 
                'Overall Recall': class_recall.mean(),
                'Overall F1': class_f1.mean(),
                'Class Acc': class_accuracy,
                'Class Precision': class_precision,
                'Class Recall': class_recall,
                'Class F1': class_f1}

    def reset(self):
        self.correct = np.zeros(self.n_classes)
