
from .pcbdataset_origin import PCBDatasetOrigin
from .pcbdataset_search import PCBDatasetSearch
from .pcbdataset_tri_origin import PCBDatasetTriOrigin


# Define pcbdataset dictionary
PCBDataset = {
    'origin': PCBDatasetOrigin,
    'search': PCBDatasetSearch,
    'tri_origin': PCBDatasetTriOrigin,
}

def get_pcbdataset(method):
    return PCBDataset[method]
