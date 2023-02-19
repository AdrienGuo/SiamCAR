from .pcbdataset import PCBDataset
from .pcbdataset_tri import PCBDatasetTri

# Define pcbdataset dictionary
pcb_dataset = {
    'siamcar': PCBDataset,
    'origin': PCBDataset,
    'official_origin': PCBDataset,
    'tri_origin': PCBDatasetTri,
    'tri_127_origin': PCBDatasetTri,
    'test': PCBDataset,
    'PatternMatch_test': PCBDatasetTri
}


def get_pcbdataset(method):
    return pcb_dataset[method]
