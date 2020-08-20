# Neural Adaptive IoT Streaming Analytics with RL-Adapt

## Requirements
* Python libraries --> requirements.txt
* [ffmpeg](https://ffmpeg.org/) --> Should be installed in /usr/bin/ffmpeg.

## Important Notes
* All command should be run in the `src` folder


### 1.1 Build Pedestrian Detection
Install [RUST](https://www.rust-lang.org/)
```sh
cd src/extra/video
sh makepd.sh
ln -s ./target/release/main ./pd
``` 
* Download dataset from [MOT Challenge](https://motchallenge.net/data/MOT16.zip)
* Copy MOT16/train/MOT16-04/img1 to folder `dataset/pd `
* Copy MOT16/test/MOT16-03/img1 to folder `dataset/pd_test`
* Change `cpu_num` in `apis/adaptmodel`, default set to 4

### 1.2 Run pedestrian detection
```sh
cd src
python3 pd.py
```

### 2.1 Build Augmented Reality
* Download [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights)
* `cp yolov3-tiny.weights src/extra/yolov3-spp.weights`
* Download the ar dataset from [Google Drive](https://drive.google.com/file/d/1k9HnzfEbOxPL5qlwBA56q5gwi_E-lBAf/view?usp=sharing) and copy to path `dataset/ar`


### 2.2 Run Augmented Reality
```sh
cd src
python3 ar.py
```

### 3.1 Build Distributed-Top-K
* Download the logk dataset from [Google Drive](https://drive.google.com/drive/folders/14eeDxWjzuJIV8W_jFYpSf7PrV1y68qy1?usp=sharing) and copy to path `dataset/logk` and `dataset/logk_test`

### 3.2 Run Distributed-Top-K
* edit `logk.py`
```sh
python3 logk.py
```

### 4 Runtime Test
* see runtime [README](./src/runtime/README.md)