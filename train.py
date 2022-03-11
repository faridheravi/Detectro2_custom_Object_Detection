from types import ClassMethodDescriptorType
from detectron2.utils.logger import setup_logger

setup_logger()


from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os, json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle

from utils import *

from detectron2.data import MetadataCatalog


config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
checkpoint_url  = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

output_dir = "./output/ObjectDetection2"
num_classes = 1

device = "cuda"

train_dataset_name = "LP_train"
train_images_path = "train"
train_json_annot_path = "train.json"


test_dataset_name = "LP_test"
test_images_path = "test"
test_json_annot_path = "test.json"

cfg_save_path = "LP_cfg.pickle"

#########################################

register_coco_instances(name = train_dataset_name, metadata={},
json_file = train_json_annot_path, image_root=train_images_path)

register_coco_instances(name = test_dataset_name, metadata={},
json_file = test_json_annot_path, image_root=test_images_path)

#keypoint_names = ['O', 'x', 'y', 'z']
#keypoint_flip_map = [['O' , 'x'], ['O', 'y'], ['O', 'z']]
#from detectron2.data import MetadataCatalog
#classes = MetadataCatalog.get("KP_train").thing_classes = ["Screw"]
#print(classes)


#MetadataCatalog.get("KP_train").thing_classes = ["Screw"]
#MetadataCatalog.get("KP_train").thing_dataset_id_to_contiguous_id = {3:0}
#MetadataCatalog.get("KP_train").keypoint_names =keypoint_names
#MetadataCatalog.get("KP_train").keypoint_flip_map = keypoint_flip_map
#MetadataCatalog.get("KP_train").evaluater_type = "coco"


#plot_samples(dataset_name=train_dataset_name, n=2)


########################

def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':

  main()
      