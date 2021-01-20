import os

from .ship import register_ship
from .voc2007 import register_voc
from .air_sar1 import register_air_sar1
from .air_sar2 import register_air_sar2
from .coco_class import register_coco_class

from detectron2.data import MetadataCatalog
from itertools import product


def register_all_coco_class(root):
    for name, split in product(["coco"], ["train2014", "val2014"]):
        dataset_name = name + split
        register_coco_class(dataset_name, 
                            os.path.join(root, name, f"annotations/instances_{split}.json"), 
                            os.path.join(root, name, split), 
                            ["person", "car"])
        MetadataCatalog.get(dataset_name).evaluator_type = "coco_class"


def register_all_ship(root):
    for name, split in product(["Ship", "Ship_Part1", "SSDD"], ["trainval", "test"]):
        dataset_name = name + split
        dirname = os.path.join(root, name)
        register_ship(dataset_name, dirname, split)
        MetadataCatalog.get(dataset_name).evaluator_type = "ship"
        
def register_all_voc(root):
    for name, split in product(["VOC2007"], ["trainval", "test"]):
        dataset_name = name + split
        dirname = os.path.join(root, name)
        register_voc(dataset_name, dirname, split)
        MetadataCatalog.get(dataset_name).evaluator_type = "voc_class"  
        
def register_all_air_sar1(root):
    for name, split in product(["AIR-SARShip-1.0"], ["trainval", "test"]):
        dataset_name = name + split
        dirname = os.path.join(root, name)
        register_air_sar1(dataset_name, dirname, split)
        MetadataCatalog.get(dataset_name).evaluator_type = "ship"
        
def register_all_air_sar2(root):
    for name, split in product(["AIR-SARShip-2.0"], ["trainval", "test"]):
        dataset_name = name + split
        dirname = os.path.join(root, name)
        register_air_sar2(dataset_name, dirname, split)
        MetadataCatalog.get(dataset_name).evaluator_type = "ship"  
        
def register_all_hrsid(root):
    from detectron2.data.datasets import register_coco_instances
    for name, split in product(["HRSID"], ["train2017", "test2017", "train_test2017"]):
        dataset_name = name + split
        dirname = os.path.join(root, name)
        json_file = os.path.join(dirname, "annotations")
        image_folder = os.path.join(dirname, "JPEGImages")

        register_coco_instances(dataset_name, {}, 
                                os.path.join(json_file, f"{split}.json"), 
                                image_folder), 

_root = os.getenv("DETECTRON2_DATASETS", "/home/jtli/SAR/dataset")
register_all_coco_class(_root)
register_all_ship(_root)
register_all_voc(_root)
register_all_air_sar1(_root)
register_all_air_sar2(_root)
register_all_hrsid(_root)