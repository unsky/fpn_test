"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr




DEBUG = False


class AssignRoisOperator(mx.operator.CustomOp):
    def __init__(self):
        super(AssignRoisOperator, self).__init__()


    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()
        score = in_data[1].asnumpy()    
        k0=4
        w = rois[:,3]-rois[:,1]
        h = rois[:,4]-rois[:,2]
        s = w * h
        s[s<=0] = 1e-6
        layer_indexs = np.floor(k0+np.log2(np.sqrt(s)/63))
        layer_indexs[layer_indexs<3] = 3
        layer_indexs[layer_indexs>5] = 5

        rois3 = []
        rois4 = []
        rois5 = []

        score3 = []
        score4 = []
        score5 = []
        for i in range(len(layer_indexs)):
            if layer_indexs[i] == 3:
                rois3.append(rois[i])
                score3.append(score[i][0])
            if layer_indexs[i] == 4:
                rois4.append(rois[i])
                score4.append(score[i][0])
            if layer_indexs[i] == 5:
                score5.append(rois[i][0])
                rois5.append(rois[i])



        post_nms_topN = 300
  
        score3 = np.array(score3)
        score4 = np.array(score4)        
        score5 = np.array(score5)

        keep3 = np.where(score3>=0)
        keep4 = np.where(score4>=0)
        keep5 = np.where(score5>=0)

        rois3= np.array(rois3)
        rois4= np.array(rois4)
        rois5= np.array(rois5) 

        if len(rois5)== 0:
            rois5 = rois4[0:1]
            score5 = score4[0:1]
            keep5 = np.where(score5>=0)
        if len(rois3)==0:
            rois3 = rois4[0:1]
            score3 = score4[0:1]
            keep3 = np.where(score3>=0)
            
  

#########################


        if len(keep3[0]) > post_nms_topN:
            order = score3.ravel().argsort()[::-1]
            order = order[0:post_nms_topN]
            rois3=rois3[order]
        if len(keep4[0]) > post_nms_topN:
            order = score4.ravel().argsort()[::-1]
            order = order[0:post_nms_topN]
            rois4=rois4[order]
        if len(keep5[0]) > post_nms_topN:
            order = score5.ravel().argsort()[::-1]
            order = order[0:post_nms_topN]
            rois5=rois5[order]
##################

        if len(keep3[0]) < post_nms_topN:
            pad = npr.choice(keep3[0], size=post_nms_topN - len(keep3[0]))
            keep_ = np.hstack((keep3[0], pad))
            rois3 = rois3[keep_, :]
        if len(keep4[0]) < post_nms_topN:
            pad = npr.choice(keep4[0], size=post_nms_topN - len(keep4[0]))
            keep_ = np.hstack((keep4[0], pad))
            rois4 = rois4[keep_, :]
        if len(keep5[0]) < post_nms_topN:
            pad = npr.choice(keep5[0], size=post_nms_topN - len(keep5[0]))
            keep_ = np.hstack((keep5[0], pad))
            rois5 = rois5[keep_, :]
        
        rois = np.vstack((rois3,rois4,rois5))
        for ind, val in enumerate([rois]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
    


@mx.operator.register('assign_rois')
class AssignRoisProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AssignRoisProp, self).__init__(need_top_grad=False)


    def list_arguments(self):
        return ['rois','score']

    def list_outputs(self):
        return ['rois']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        score_shape = in_shape[1]
        return [rois_shape,score_shape], \
               [rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignRoisOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
