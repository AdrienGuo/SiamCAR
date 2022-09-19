import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.onnx
import onnx
import onnxruntime

from pysot.core.config import cfg
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
import os
import argparse

parser = argparse.ArgumentParser(description='convert onnx')
parser.add_argument('--snapshot', type=str, default='/tf/SiamCAR/snapshot/checkpoint_e999.pth',help='snapshot')
parser.add_argument('--onnx_dir',  type=str, default='/tf/SiamCAR/onnx/SiamCAR_test.onnx',help='onnx path')
parser.add_argument('--cfg', default="/tf/SiamCAR/experiments/siamcar_r50/config_test.yaml", type=str,help='config path')
args = parser.parse_args()


def remove_initializer_from_input(in_module, out_module):
    model = onnx.load(in_module)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, out_module)
    

def onnx_export(model, input, save_dir, opset, parallel=False):
    
    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      save_dir,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=opset,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['template','search'],  # the model's input names
                      output_names=['cls','loc','cen'],)  # the model's output names
    '''
    if parallel:
        torch.onnx.export(model.module,  # model being run
                          input,  # model input (or a tuple for multiple inputs)
                          save_dir,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=opset,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['template','search'],  # the model's input names
                          output_names=['cls','loc','cen'],  # the model's output names
     '''

class ConvertModel(nn.Module):
    def __init__(self, model):
        super(ConvertModel, self).__init__()
        self.model = model
                          

    def forward(self, template, search):
        zf = self.model.backbone(template)
        xf = self.model.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.model.neck(zf)
            xf = self.model.neck(xf)

        features = self.model.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.model.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.model.down(features)

        cls, loc, cen = self.model.car_head(features)
        
        return cls, loc, cen


def convert_onnx(opset=10):
    with torch.no_grad():
        cfg.merge_from_file(args.cfg)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model0 = ModelBuilder()
        # load model
        model0 = load_pretrain(model0, args.snapshot).to(device).eval()
        model = ConvertModel(model0)

        template = torch.Tensor(np.random.rand(1,3,127,127))
        search = torch.Tensor(np.random.rand(1,3,1200,1200))
        template = template.to(device)
        search = search.to(device)


        #5. 得到 pytorch 輸出的結果
        netg_out = model(template,search)
        print("out:",netg_out[0].shape)
        print("out:",netg_out[1].shape)
        print("out2:",netg_out[2].shape)

        #6. Export the model
        onnx_export(model, (template,search), args.onnx_dir, opset, False)

        #7. 清除一些模型轉換後的warning msg
        remove_initializer_from_input(args.onnx_dir, args.onnx_dir)

        # 測試步驟
        # 1. 讀 onnx
        onnx_model = onnx.load(args.onnx_dir)
        # 2. 確認有沒有什麼 error
        onnx.checker.check_model(onnx_model)

        
if __name__ == '__main__': 
   
    convert_onnx()