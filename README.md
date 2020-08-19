# Neural Adaptive IoT Streaming Analytics with RL-Adapt

## Refinement proceeds ...


## Requirements
* Python libraries --> requirements.txt
* [ffmpeg](https://ffmpeg.org/) --> Should be installed in /usr/bin/ffmpeg.

## Important Notes
* All command should be run in the `src` folder


#### Build Pedestrian Detection
Install [RUST](https://www.rust-lang.org/)
```sh
cd src/extra/video
sh makepd.sh
ln -s ./target/release/main ./pd
``` 
* Download dataset from [Mot Challenge](https://motchallenge.net/data/MOT16.zip)
* Copy MOT16/train/MOT16-04/img1 to folder A
* Change the dataset path to folder A in `src/pd.py`

#### Run pedestrian detection
```sh
cd src
python3 pd.py
```

#### Build Augmented Reality
* Download [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
* `cp yolov3-tiny.weights src/extra/yolov3-tiny.weights`
* Download the ar dataset from [Google Drive](https://drive.google.com/drive/folders/14eeDxWjzuJIV8W_jFYpSf7PrV1y68qy1?usp=sharing) and copy to path B
* Change the dataset path to folder B in `src/ar.py`



#### Run Augmented Reality
```sh
cd src
python3 ar.py
```

#### Build Distributed-Top-K
* Download the ar dataset from [Google Drive](https://drive.google.com/drive/folders/14eeDxWjzuJIV8W_jFYpSf7PrV1y68qy1?usp=sharing) and copy to path B
* Change the dataset path to folder B in `src/logk.py`

#### Run Distributed-Top-K
* edit `logk.py`
```sh
python3 logk.py
```

#### Test Runtime
* start server in `runtime/client`
* start client in `runtime/server`
* implement traffic control in `runtime/traffic_control`
