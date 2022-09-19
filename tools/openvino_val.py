import time
import cv2 
import argparse
import os
import numpy as np
from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.datasets.testdata import ImageFolderWithSelect
from openvino.inference_engine import IECore, IENetwork
from pysot.tracker.siamcar_tracker_openvino_onnx import SiamCARTracker

def run_app():
    """
    Run Object Detection Application
    :return:
    """
    cfg.merge_from_file(arguments.config)
    # hp_search
    hp = {'lr': 0.4, 'penalty_k':0.2, 'window_lr':0.3}

    # load dataset
    dataset = ImageFolderWithSelect(arguments.dataset)
    
    # Load Network
    OpenVinoNetwork = IENetwork(model=arguments.model_xml, weights=arguments.model_bin)
    
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)
    
    # Create Executable Network
    OpenVinoExecutable = OpenVinoIE.load_network(network=OpenVinoNetwork, device_name=arguments.target_device)
    
    # build tracker
    tracker = SiamCARTracker(OpenVinoExecutable,arguments.target_device,'openvino', cfg.TRACK)

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


                model_path = os.path.join('results','openvino')
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                    
                # save results
                temp = name+"__"+str(classid)+"__"+str(count)
                result_path = os.path.join(model_path, '{}.txt'.format(name+"__"+str(classid)+"__"+str(count)))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+"="+'\n')

global arguments
global number_of_async_req

"""
Entry Point of Application
"""
if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Basic OpenVINO Example with SiamCAR')
    parser.add_argument('--model-xml',default='/tf/SiamCAR/onnx/SiamCAR.xml',help='XML File')
    parser.add_argument('--model-bin',default='/tf/SiamCAR/onnx/SiamCAR.bin',help='BIN File')
    parser.add_argument('--target-device', default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('--input-type', default='image', help='Type of Input: image, video, cam')
    parser.add_argument('--dataset', default='/tf/SiamCAR/testing_dataset/DATA/', help='Path to origin image')
    parser.add_argument('--async',action="store_true", default=False, help='Run Async Mode')
    parser.add_argument('--config', type=str, default='/tf/SiamCAR/experiments/siamcar_r50/config_test.yaml',
        help='config file')
    parser.add_argument('--request-number', default=1, help='Number of Requests')

    global arguments
    arguments = parser.parse_args()

    global number_of_async_req
    number_of_async_req = int(arguments.request_number)

    print('WARNING: No Argument Control Done, You Can GET Runtime Errors')
   
    run_app()
