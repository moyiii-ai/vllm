#!/bin/bash

# Check if the script is running with root privileges
# If not, prompt the user to run with sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with sudo, for example: sudo $0"
    exit 1
fi

# Define the list of PCIe devices to check
devices=(
    "be:03.0"
    "be:04.0"
    "bc:01.0"
)

# Check link status for each device
for device in "${devices[@]}"; do
    echo "Checking link status for device $device..."
    
    # Execute lspci command and capture output
    result=$(lspci -vvv -s "$device" | grep LnkSta)
    
    # Check if command execution was successful
    if [ -z "$result" ]; then
        echo "Error: Failed to retrieve link status for device $device"
        continue
    fi
    
    # Output raw result
    echo "Raw output: $result"
    
    # For the first two devices, verify if speed is 16GT/s
    if [ "$device" = "be:03.0" ] || [ "$device" = "be:04.0" ]; then
        if echo "$result" | grep -q "16GT/s"; then
            echo "Verification result: Device $device link speed is 16GT/s (as expected)"
        else
            echo "Verification result: Device $device link speed is not 16GT/s (not as expected)"
        fi
    fi
    
    echo "----------------------------------------"
done