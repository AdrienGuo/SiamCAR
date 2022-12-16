from .pcbdataset import PCBDataset
from .pcbdataset_tri import PCBDatasetTri

# Define pcbdataset dictionary
pcbdataset_ = {
    'origin': PCBDataset,
    'search': PCBDataset,
    'official': PCBDataset,
    'tri_origin': PCBDatasetTri
}

def get_pcbdataset(method):
    return pcbdataset_[method]
