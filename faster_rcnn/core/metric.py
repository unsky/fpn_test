# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx
import numpy as np


def get_rpn_names():
    pred = ['rpn_cls_prob/p3','rpn_bbox_loss/p3', 'rpn_cls_prob/p4', 'rpn_bbox_loss/p4','rpn_cls_prob/p5', 'rpn_bbox_loss/p5']

    label = ['rpn_label/p3','rpn_label/p4','rpn_label/p5','rpn_bbox_target/p3', 'rpn_bbox_target/p4', 'rpn_bbox_target/p5','rpn_bbox_weight/p3','rpn_bbox_weight/p4','rpn_bbox_weight/p5']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob/p3', 'rcnn_bbox_loss/p3','rcnn_label/p3','rcnn_cls_prob/p4', 'rcnn_bbox_loss/p4','rcnn_label/p4','rcnn_cls_prob/p5', 'rcnn_bbox_loss/p5','rcnn_label/p5']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']

    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label

class p3RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p3RPNAccMetric, self).__init__('RPNAcc/p3')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('rpn_cls_prob/p3')]
        label = labels[self.label.index('rpn_label/p3')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class p4RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p4RPNAccMetric, self).__init__('RPNAcc/p4')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('rpn_cls_prob/p4')]
        label = labels[self.label.index('rpn_label/p4')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class p5RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p5RPNAccMetric, self).__init__('RPNAcc/p5')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob/p5')]
        label = labels[self.label.index('rpn_label/p5')]
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)




class p3RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p3RPNLogLossMetric, self).__init__('RPNLogLoss/p3')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob/p3')]
        label = labels[self.label.index('rpn_label/p3')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]
class p4RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p4RPNLogLossMetric, self).__init__('RPNLogLoss/p4')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob/p4')]
        label = labels[self.label.index('rpn_label/p4')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]
class p5RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p5RPNLogLossMetric, self).__init__('RPNLogLoss/p5')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob/p5')]
        label = labels[self.label.index('rpn_label/p5')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class p3RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p3RPNL1LossMetric, self).__init__('RPNL1Loss/p3')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss/p3')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p3')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
class p4RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p4RPNL1LossMetric, self).__init__('RPNL1Loss/p4')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss/p4')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p4')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class p5RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(p5RPNL1LossMetric, self).__init__('RPNL1Loss/p5')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss/p5')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label/p5')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst




class p3RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p3RCNNAccMetric, self).__init__('RCNNAcc/p3')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p3')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p3')]
        else:
            label = labels[self.label.index('rcnn_label/p3')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')


        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
        print "p3label:",label
        print 'p3pred:',pred_label
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class p3RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p3RCNNLogLossMetric, self).__init__('RCNNLogLoss/p3')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p3')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p3')]
        else:
            label = labels[self.label.index('rcnn_label/p3')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class p3RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p3RCNNL1LossMetric, self).__init__('RCNNL1Loss/p3')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss/p3')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label/p3')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label/p3')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label/p3')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class p4RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p4RCNNAccMetric, self).__init__('RCNNAcc/p4')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p4')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p4')]
        else:
            label = labels[self.label.index('rcnn_label/p4')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')


        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
        print "p4label:",label
        print 'p4pred:',pred_label
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class p4RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p4RCNNLogLossMetric, self).__init__('RCNNLogLoss/p4')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p4')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p4')]
        else:
            label = labels[self.label.index('rcnn_label/p4')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class p4RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p4RCNNL1LossMetric, self).__init__('RCNNL1Loss/p4')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss/p4')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label/p4')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label/p4')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label/p4')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

###################

class p5RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p5RCNNAccMetric, self).__init__('RCNNAcc/p5')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p5')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p5')]
        else:
            label = labels[self.label.index('rcnn_label/p5')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]
        print "p5label:",label
        print 'p5pred:',pred_label
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class p5RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p5RCNNLogLossMetric, self).__init__('RCNNLogLoss/p5')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob/p5')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label/p5')]
        else:
            label = labels[self.label.index('rcnn_label/p5')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class p5RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(p5RCNNL1LossMetric, self).__init__('RCNNL1Loss/p5')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss/p5')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label/p5')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label/p5')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label/p5')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
#################



# class RCNNL1LossMetric(mx.metric.EvalMetric):
#     def __init__(self, cfg):
#         super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
#         self.e2e = cfg.TRAIN.END2END
#         self.ohem = cfg.TRAIN.ENABLE_OHEM
#         self.pred, self.label = get_rcnn_names(cfg)

#     def update(self, labels, preds):
#         bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
#         if self.ohem:
#             label = preds[self.pred.index('rcnn_label')].asnumpy()
#         else:
#             if self.e2e:
#                 label = preds[self.pred.index('rcnn_label')].asnumpy()
#             else:
#                 label = labels[self.label.index('rcnn_label')].asnumpy()

#         # calculate num_inst (average on those kept anchors)
#         num_inst = np.sum(label != -1)

#         self.sum_metric += np.sum(bbox_loss)
#         self.num_inst += num_inst

# class RCNNLogLossMetric(mx.metric.EvalMetric):
#     def __init__(self, cfg):
#         super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
#         self.e2e = cfg.TRAIN.END2END
#         self.ohem = cfg.TRAIN.ENABLE_OHEM
#         self.pred, self.label = get_rcnn_names(cfg)

#     def update(self, labels, preds):
#         pred = preds[self.pred.index('rcnn_cls_prob')]
#         if self.ohem or self.e2e:
#             label = preds[self.pred.index('rcnn_label')]
#         else:
#             label = labels[self.label.index('rcnn_label')]

#         last_dim = pred.shape[-1]
#         pred = pred.asnumpy().reshape(-1, last_dim)
#         label = label.asnumpy().reshape(-1,).astype('int32')

#         # filter with keep_inds
#         keep_inds = np.where(label != -1)[0]
#         label = label[keep_inds]
#         cls = pred[keep_inds, label]

#         cls += 1e-14
#         cls_loss = -1 * np.log(cls)
#         cls_loss = np.sum(cls_loss)
#         self.sum_metric += cls_loss
#         self.num_inst += label.shape[0]

# class RCNNAccMetric(mx.metric.EvalMetric):
#     def __init__(self, cfg):
#         super(RCNNAccMetric, self).__init__('RCNNAcc')
#         self.e2e = cfg.TRAIN.END2END
#         self.ohem = cfg.TRAIN.ENABLE_OHEM
#         self.pred, self.label = get_rcnn_names(cfg)

#     def update(self, labels, preds):
#      #   print preds
#         pred = preds[self.pred.index('rcnn_cls_prob')]
#         if self.ohem or self.e2e:
#             label = preds[self.pred.index('rcnn_label')]
#         else:
#             label = labels[self.label.index('rcnn_label')]

#         last_dim = pred.shape[-1]
#         pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
#         label = label.asnumpy().reshape(-1,).astype('int32')

#         # filter with keep_inds
#         keep_inds = np.where(label != -1)
#         pred_label = pred_label[keep_inds]
#         label = label[keep_inds]
#         print 'pred:',pred_label
#         print 'label:',label
#         self.sum_metric += np.sum(pred_label.flat == label.flat)
#         self.num_inst += len(pred_label.flat)