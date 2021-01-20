# import some common libraries
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

def load_air_sar2_instances(dirname: str, split: str):
    """
    Load air SAR 2.0 detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "JPEGImages"
        split (str): one of "train", "test"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "AIR-SARShip-2.0-xml", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "AIR-SARShip-2.0-data", fileid + ".tiff")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": 1000,
            "width": 1000,
        }
        instances = []

        for obj in tree.find('objects').findall("object"):
            cls = obj.find('possibleresult').find('name').text
            bbox = obj.find("points")
            xmin = ymin = float(10000)
            xmax = ymax = 0.                
            for bbox_node in bbox.findall('point'):
                coor = bbox_node.text.split(',')
                x, y = map(float, coor)
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                xmax = max(xmax, x)
                ymax = max(ymax, y) 
                
            instances.append(
                {"category_id": 0, "bbox": [xmin, ymin, xmax, ymax], "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_air_sar2(name, dirname, split):
    DatasetCatalog.register(name,
                            lambda: load_air_sar2_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=['ship'],
                                  dirname=dirname,
                                  yeas=2007,
                                  split=split)