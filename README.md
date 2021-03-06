# Real time segmentation for occupancy grid generation
Repository for the Paper "Occupancy Grid Generation with Dynamic Obstacle Segmentation in Stereo Images".

*Semantic segmentation results of various trained deep networks on Mapillary Vistas Dataset:*
<img src="imgs/Segmentation_examples.jpg" width="100%">

# OpenTaganrog Dataset 
We propose custom **OpenTaganrog dataset**: **[link to download](https://yadi.sk/d/jjExD7_YiTKqbg?w=1)**

Custom  OpenTaganrog  dataset consists of rosbag-archives. They include  point  clouds  from Velodyne  VLP32C  LiDAR,  left  and  right  stereo  camera  images,  and  vehicle  position  and  orientation  from  inertial navigation system.  The recording  was  made  from  a  car  in  urban  traffic  conditions in Taganrog City, Russia. Odometry was obtained from INS Atlans-C in RTK mode withan accuracy of 0.01m. The dataset contains more than 98000 unlabeled  images  from  a  stereo  camera  with  a  resolution  of 1924x1084.

*Image samples from OpenTaganrog dataset:*
<img src="imgs/OpenTaganrog_samples.jpg" width="100%">

## Network
***Training***
```shell
    sh tools/train.sh
```

***Saving predicted masks***
```shell
    sh tools/vis.sh
```

***Visualization***
```shell
    sh vis_utils/run_vis.sh
```

***Evaluation of model***
```shell
    sh tools/eval.sh
```

***Evaluation from two directories***
```shell
    python eval/get_iou_metric.py
```