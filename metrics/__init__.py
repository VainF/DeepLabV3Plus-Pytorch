from .stream_metrics import StreamCompMetrics, StreamSegMetrics, AverageMeter

def get_metrics(num_classes):
    return StreamSegMetrics(num_classes)

def get_average_meter():
    return AverageMeter()
