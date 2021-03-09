TRUE_LABEL_DIR = "/media/cds-k/Data_2/DATASETS/Mapillary/val/1920_1080/labels"
PRED_LABEL_DIR = "../pred_images"

TRUE_FLAG_CONVERT = False
PRED_FLAG_CONVERT = False
BATCH_SIZE = 1
NUM_CLASSES = 66
PALETTE = {}
            
ID2CLASSES = {0: 'animal--bird', 1: 'animal--ground-animal', 2: 'construction--barrier--curb',
              3: 'construction--barrier--fence', 4: 'construction--barrier--guard-rail',
              5: 'construction--barrier--other-barrier', 6: 'construction--barrier--wall',
              7: 'construction--flat--bike-lane', 8: 'construction--flat--crosswalk-plain',
              9: 'construction--flat--curb-cut', 10: 'construction--flat--parking',
              11: 'construction--flat--pedestrian-area', 12: 'construction--flat--rail-track',
              13: 'construction--flat--road', 14: 'construction--flat--service-lane',
              15: 'construction--flat--sidewalk', 16: 'construction--structure--bridge',
              17: 'construction--structure--building', 18: 'construction--structure--tunnel',
              19: 'human--person', 20: 'human--rider--bicyclist', 21: 'human--rider--motorcyclist',
              22: 'human--rider--other-rider', 23: 'marking--crosswalk-zebra', 24: 'marking--general',
              25: 'nature--mountain', 26: 'nature--sand', 27: 'nature--sky', 28: 'nature--snow',
              29: 'nature--terrain', 30: 'nature--vegetation', 31: 'nature--water', 32: 'object--banner',
              33: 'object--bench', 34: 'object--bike-rack', 35: 'object--billboard', 36: 'object--catch-basin',
              37: 'object--cctv-camera', 38: 'object--fire-hydrant', 39: 'object--junction-box',
              40: 'object--mailbox', 41: 'object--manhole', 42: 'object--phone-booth', 43: 'object--pothole',
              44: 'object--street-light', 45: 'object--support--pole', 46: 'object--support--traffic-sign-frame',
              47: 'object--support--utility-pole', 48: 'object--traffic-light', 49: 'object--traffic-sign--back',
              50: 'object--traffic-sign--front', 51: 'object--trash-can', 52: 'object--vehicle--bicycle',
              53: 'object--vehicle--boat', 54: 'object--vehicle--bus', 55: 'object--vehicle--car',
              56: 'object--vehicle--caravan', 57: 'object--vehicle--motorcycle', 58: 'object--vehicle--on-rails',
              59: 'object--vehicle--other-vehicle', 60: 'object--vehicle--trailer', 61: 'object--vehicle--truck',
              62: 'object--vehicle--wheeled-slow', 63: 'void--car-mount', 64: 'void--ego-vehicle',
              65: 'void--unlabeled'}


