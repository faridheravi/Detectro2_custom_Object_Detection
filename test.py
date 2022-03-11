from detectron2.engine import DefaultPredictor

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pickle


from utils import *

cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)


cfg.MODEL.WEITHS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)



image_path = "test/IMG_20211013_104830.jpg"
videoPath  = "test/test1.avi"

on_image(image_path, predictor)
on_video(videoPath, predictor)