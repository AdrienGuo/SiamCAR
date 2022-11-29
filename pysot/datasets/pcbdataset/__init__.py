
from .pcbdataset_origin import PCBDatasetOrigin
from .pcbdataset_new import PCBDatasetNew
from .pcbdataset_tri_origin import PCBDatasetTriOrigin


# Define pcbdataset dictionary
PCBDataset = {
    'origin': PCBDatasetOrigin,
    'new': PCBDatasetNew,
    'tri_origin': PCBDatasetTriOrigin,
}

def get_pcbdataset(method):
    return PCBDataset[method]
