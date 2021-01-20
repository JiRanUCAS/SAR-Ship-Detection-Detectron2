# import some common libraries
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

def load_ship_instances(dirname: str, split: str):
    """
    Load Ship detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "JPEGImages"
        split (str): one of "train", "test"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            instances.append(
                {"category_id": 0, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_ship(name, dirname, split):
    DatasetCatalog.register(name,
                            lambda: load_ship_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=['ship'],
                                  dirname=dirname,
                                  year=2007,
                                  split=split)