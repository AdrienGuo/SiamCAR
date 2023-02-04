from .tri.official_origin import CropTri127Origin
from .tri.origin import CropTriOrigin
from .pcb.official_origin import PCBCropOfficialOrigin
from .pcb.origin import PCBCropOrigin
from .pcb.siamcar import PCBCropCAR

PCBCrop = {
    'siamcar': PCBCropCAR,
    'origin': PCBCropOrigin,
    'official_origin': PCBCropOfficialOrigin,
    'tri_origin': CropTriOrigin,
    'tri_127_origin': CropTri127Origin,
}


def get_pcb_crop(method):
    return PCBCrop[method]
