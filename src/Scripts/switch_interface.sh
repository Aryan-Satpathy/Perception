#!/bin/bash

# Switch network interface
if [ $# != 1 ]; then
    echo "Specificy inteface to switch to as an arguement: (options: ethernet or wifi)"

# Switch to Ethernet
elif [ $1 = "ethernet" ]; then
    sudo ifmetric wlp3s0 20
    sudo ifmetric enp2s0f0 10

# Switch to Wifi
elif [ $1 = "wifi" ]; then
    sudo ifmetric enp2s0f0 20
    sudo ifmetric wlp3s0 10
else
    echo "Specificy inteface to switch to as an arguement: (options: ethernet or wifi)"
fi