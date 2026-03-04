#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <add/remove>"
    exit 1
fi

cmd=$1

if [ "$cmd" != "add" ] && [ "$cmd" != "remove" ]; then
    echo "Usage: $0 <add/remove>"
    exit 1
fi

# Assign 4 NICs to 4 different netns
iface0=enp63s0f0np0
iface1=enp47s0np0
iface2=enp193s0np0
iface3=enp175s0f0np0

# Get Mellanox device names (if needed)
iface0_mlx=$(ibdev2netdev | grep $iface0 | cut -d' ' -f1)
iface1_mlx=$(ibdev2netdev | grep $iface1 | cut -d' ' -f1)
iface2_mlx=$(ibdev2netdev | grep $iface2 | cut -d' ' -f1)
iface3_mlx=$(ibdev2netdev | grep $iface3 | cut -d' ' -f1)


function add_nics_to_netns() {    
    netnsctl add h1
    netnsctl add h2
    netnsctl add h3
    netnsctl add h4

    netnsctl assign $iface0 h1
    netnsctl assign $iface1 h2
    netnsctl assign $iface2 h3
    netnsctl assign $iface3 h4

    sleep 2

    sudo ip netns exec h1 ip addr add 10.1.1.1/24 dev $iface0
    sudo ip netns exec h1 ip link set $iface0 up
    sudo ip netns exec h2 ip addr add 10.1.1.2/24 dev $iface1
    sudo ip netns exec h2 ip link set $iface1 up
    sudo ip netns exec h3 ip addr add 10.1.1.3/24 dev $iface2
    sudo ip netns exec h3 ip link set $iface2 up
    sudo ip netns exec h4 ip addr add 10.1.1.4/24 dev $iface3
    sudo ip netns exec h4 ip link set $iface3 up
}

function remove_nics_from_netns() {
    yes | netnsctl delete h1 > /dev/null 2>&1
    yes | netnsctl delete h2 > /dev/null 2>&1
    yes | netnsctl delete h3 > /dev/null 2>&1
    yes | netnsctl delete h4 > /dev/null 2>&1

    sleep 2

    sudo ip link set $iface0 up
    sudo ip link set $iface1 up
    sudo ip link set $iface2 up
    sudo ip link set $iface3 up
}

if [ "$cmd" == "add" ]; then
    add_nics_to_netns
elif [ "$cmd" == "remove" ]; then
    remove_nics_from_netns
fi

