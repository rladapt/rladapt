## How to Run

1. Modify the PORT and SERVER_HOST in the two files separately.
2. Run `python3 rladapt/src/localserver.py` to start the rladapt server.
3. Run `python3 traffic_control/main.py` at Pi4 to start the network traffic control.
4. Run `python3 server.py` at server platform to start the transformation server.
5. Run the following command in the PI4 shell to start the transformation client.
```shell
~$ python3 client.py
~$ cd traffic_control
~$ ./main.sh
```
6. when the traffic_control script stopped, use Ctrl+C to stop the transformation server. Result will be saved at result.csv.

