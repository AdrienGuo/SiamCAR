from .pcb_crop_official import PCBCropOfficial
from .pcb_crop_origin import PCBCropOrigin
from .pcb_crop_search import PCBCropSearch
from .pcb_crop_tri_origin import PCBCropTriOrigin

PCBCrop = {
    'official': PCBCropOfficial,
    'search': PCBCropSearch,
    'origin': PCBCropOrigin,
    'tri_origin': PCBCropTriOrigin
}

def get_pcb_crop(method):
    return PCBCrop[method]
