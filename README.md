# RL-Adapt source code
## refinement proceeds ...


## requirements
install if the code does not work

#### build pedestrian detection
```sh
cd video
sh makepd.sh
cd ../src
ln -s ../video/target/release/main pd
``` 

#### run pedestrian detection
```sh
cd RLframework
# edit pd.py
python3 pd.py
```

#### build augmented reality
* download `yolov3-tiny.cfg`, `yolov3-tiny.weights`

#### run augmented reality
* edit path in `RLframework/ar_interface.py`
* edit `ar.py`
```sh
python3 ar.py
```

#### build distributed-topk
* download the dataset mentioned in paper
* use makeslice() in `RLframework/distlogk_interface.py`

#### run distributed-topk
* edit `logk.py`
```sh
python3 logk.py
```

#### test runtime
* start server in `runtime/client`
* start client in `runtime/server`
* implement traffic control in `runtime/traffic_control`
