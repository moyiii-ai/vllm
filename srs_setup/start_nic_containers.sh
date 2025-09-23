#!/bin/bash
set -e

CTN0_NAME="ctn0"
CTN1_NAME="ctn1"

CTN0_GPU="0"
CTN0_NIC="enp47s0np0"
CTN0_IP="10.1.1.1/24"

CTN1_NIC="enp193s0np0"
CTN1_IP="10.1.1.2/24"

IMAGE="vllm-with-nettools-rdma:v0.10.1"
WORKDIR="/vllm-workspace"

# Run containers
docker run -dit \
  --name $CTN0_NAME \
  --cpuset-mems=0 \
  --cpuset-cpus=0-15 \
  --gpus "device=$CTN0_GPU" \
  --network=none \
  --cap-add=NET_ADMIN \
  -w $WORKDIR \
  --entrypoint sleep \
  $IMAGE infinity

CTN0_PID=$(docker inspect -f '{{.State.Pid}}' $CTN0_NAME)

sudo ip link set $CTN0_NIC netns $CTN0_PID
sudo nsenter -t $CTN0_PID -n ip link set $CTN0_NIC up
sudo nsenter -t $CTN0_PID -n ip addr add $CTN0_IP dev $CTN0_NIC

docker run -dit \
  --name $CTN1_NAME \
  --cpuset-mems=1 \
  --cpuset-cpus=16-31 \
  --network=none \
  --cap-add=NET_ADMIN \
  -w $WORKDIR \
  --entrypoint sleep \
  $IMAGE infinity

CTN1_PID=$(docker inspect -f '{{.State.Pid}}' $CTN1_NAME)

sudo ip link set $CTN1_NIC netns $CTN1_PID
sudo nsenter -t $CTN1_PID -n ip link set $CTN1_NIC up
sudo nsenter -t $CTN1_PID -n ip addr add $CTN1_IP dev $CTN1_NIC

# Test container-to-container connectivity
echo "Testing connectivity from $CTN0_NAME ($CTN0_IP) to $CTN1_NAME ($CTN1_IP)..."
sudo nsenter -t $CTN0_PID -n ping -c 3 ${CTN1_IP%/*}

# --- Setup veth + NAT for public network access ---
# Container 0
sudo ip link add veth0-host type veth peer name veth0-ctn
sudo ip link set veth0-ctn netns $CTN0_PID
sudo ip addr add 10.10.0.1/24 dev veth0-host
sudo ip link set veth0-host up
sudo nsenter -t $CTN0_PID -n ip addr add 10.10.0.2/24 dev veth0-ctn
sudo nsenter -t $CTN0_PID -n ip link set veth0-ctn up
sudo nsenter -t $CTN0_PID -n ip route add default via 10.10.0.1

# Container 1
sudo ip link add veth1-host type veth peer name veth1-ctn
sudo ip link set veth1-ctn netns $CTN1_PID
sudo ip addr add 10.10.1.1/24 dev veth1-host
sudo ip link set veth1-host up
sudo nsenter -t $CTN1_PID -n ip addr add 10.10.1.2/24 dev veth1-ctn
sudo nsenter -t $CTN1_PID -n ip link set veth1-ctn up
sudo nsenter -t $CTN1_PID -n ip route add default via 10.10.1.1

# Enable NAT on host for outbound internet access
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o enp3s0f0np0 -s 10.10.0.0/16 -j MASQUERADE

echo "veth + NAT setup complete. Containers should now reach public network."