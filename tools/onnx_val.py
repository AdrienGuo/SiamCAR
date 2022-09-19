import time
import torch
import cv2 
import argparse
import os
import numpy as np
import onnxruntime
from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.datasets.testdata import ImageFolderWithSelect
from pysot.tracker.siamcar_tracker_openvino_onnx import SiamCARTracker

def run_app():
    """
    Run Object Detection Application
    :return:
    """
    cfg.merge_from_file(args.config)
    # hp_search
    hp = {'lr': 0.4, 'penalty_k':0.2, 'window_lr':0.3}

    # load dataset
    dataset = ImageFolderWithSelect(args.dataset)
    
    # Load Network
    device = torch.device(args.device)
   
    if args.device != 'cpu':
        onet_session = onnxruntime.InferenceSession(args.model,providers=[ 'CUDAExecutionProvider'])
    else:
        onet_session = onnxruntime.InferenceSession(args.model,providers=[ 'CPUExecutionProvider'])
    
    print("providers:",onet_session.get_providers())
    
    # build tracker
    tracker = SiamCARTracker(onet_session,args.device, 'onnx',cfg.TRACK)

    frame_count = 0
    for idx, (_,annotation,path,check) in enumerate(dataset):
        temp = path.split('/')
        name = temp[5]
        count = 0
        print("name:",name)
        for i in range (len(annotation)):
            annotation[i][1]= float(annotation[i][1])
            annotation[i][2]= float(annotation[i][2])
            annotation[i][3]= float(annotation[i][3])
            annotation[i][4]= float(annotation[i][4])
            
            pred_bboxes = []

            #init
            count +=1
            classid = annotation[i][0]
            
            if annotation[i][0]!=26:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if check==0:
                    gt_bbox = [annotation[i][1]*img.shape[1],annotation[i][2]*img.shape[0],annotation[i][3]*img.shape[1],annotation[i][4]*img.shape[0]]
                    gt_bbox = [gt_bbox[0]-gt_bbox[2]/2,gt_bbox[1]-gt_bbox[3]/2,gt_bbox[2],gt_bbox[3]]
                else:
                    gt_bbox = [annotation[i][1],annotation[i][2],annotation[i][3]-annotation[i][1],annotation[i][4]-annotation[i][2]]
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-w/2, cy-h/2, w, h]

                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)

                #track
                outputs = tracker.track(img,hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)


                model_path = os.path.join('results','onnx')
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                    
                # save results
                temp = name+"__"+str(classid)+"__"+str(count)
                result_path = os.path.join(model_path, '{}.txt'.format(name+"__"+str(classid)+"__"+str(count)))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+"="+'\n')



"""
Entry Point of Application
"""
if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Basic Onnx Example with SiamCAR')
    parser.add_argument('--model',default='/tf/SiamCAR/onnx/SiamCAR.onnx',help='onnx File')
    parser.add_argument('--device', default='cpu',
                        help='Target Plugin: cpu, cuda:0')
    parser.add_argument('--dataset', default='/tf/SiamCAR/testing_dataset/DATA/', help='Path to origin image')
    parser.add_argument('--config', type=str, default='/tf/SiamCAR/experiments/siamcar_r50/config_test.yaml',
        help='config file')

    args = parser.parse_args()
    run_app()
