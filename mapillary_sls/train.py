from pathlib import Path
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.datasets.generic_dataset import ImagesFromList
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mapillary_sls.utils.visualize import denormalize, visualize_triplets
from mapillary_sls.utils.eval import download_msls_sample


