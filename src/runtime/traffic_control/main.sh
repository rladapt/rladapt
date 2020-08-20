function main(){
	echo "start shaping"
	trap "stop" SIGHUP SIGINT SIGTERM
	sudo tc qdisc add dev lo root handle 1: htb default 12
	sudo tc filter add dev lo protocol ip parent 1: prio 1 u32 match ip dst 127.0.0.1 flowid 1:12
	sudo tc class add dev lo parent 1:1 classid 1:12 htb rate 20mbit ceil 20mbit
	 sudo tc qdisc add dev lo parent 1:12 handle 12:  netem delay 50ms 10ms distribution normal
	 echo "set to 20 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 18mbit ceil 18mbit
	 echo "set to 18 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 16mbit ceil 16mbit
	 echo "set to 16 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 14mbit ceil 14mbit
	 echo "set to 14 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 12mbit ceil 12mbit
	 echo "set to 12 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 10mbit ceil 10mbit
	 echo "set to 10 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 8mbit ceil 8mbit
	 echo "set to 8 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 6mbit ceil 6mbit
	 echo "set to 6 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 4mbit ceil 4mbit
	 echo "set to 4 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 2mbit ceil 2mbit
	 echo "set to 2 mbit"
	sleep 20

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 4mbit ceil 4mbit
	 echo "set to 4 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 6mbit ceil 6mbit
	 echo "set to 6 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 8mbit ceil 8mbit
	 echo "set to 8 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 10mbit ceil 10mbit
	 echo "set to 10 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 12mbit ceil 12mbit
	 echo "set to 12 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 14mbit ceil 14mbit
	 echo "set to 14 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 16mbit ceil 16mbit
	 echo "set to 16 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 18mbit ceil 18mbit
	 echo "set to 18 mbit"
	sleep 10

	sudo tc class change dev lo parent 1:1 classid 1:12 htb rate 20mbit ceil 20mbit
	 echo "set to 20 mbit"
	sleep 10

	stop
}
function stop(){
	echo "stopped" 
	sudo tc qdisc del dev lo root
	 exit 0 
}
main