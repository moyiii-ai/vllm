#!/bin/bash

# ==============================================================================
# pcie-speed.sh - A script to control PCIe link speed for a given device.
#
# Description:
#   This script forces a PCIe device's link speed to a specific generation
#   (e.g., Gen4) or resets it to the maximum speed supported by the hardware.
#   It automatically locates the required PCI Express capability registers.
#
# Usage:
#   sudo ./pcie-speed.sh <bdf> <action>
#
# Arguments:
#   bdf:    The Bus:Device.Function address of the target PCIe device
#           (e.g., 01:00.0 or ac:04.0).
#   action: 'gen4' to force 16.0 GT/s, or 'reset' to allow negotiation
#           up to the device's maximum supported speed.
#
# Requirements:
#   - pciutils (lspci, setpci) must be installed.
#   - Must be run with root privileges (sudo).
# ==============================================================================

# --- Configuration and Constants ---
# Color codes for output messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Standard PCIe register offsets relative to the Express Capability header
LINK_CAP_REG_OFFSET=0x0C      # Link Capabilities Register (for max speed)
LINK_CTRL_REG_OFFSET=0x10     # Link Control Register (for retraining)
LINK_CTRL_2_REG_OFFSET=0x30   # Link Control 2 Register (for target speed)

# --- Function Definitions ---

# Prints the script's usage instructions and exits.
print_usage() {
    echo -e "${YELLOW}Usage: sudo $0 <bdf> <gen4|reset>${NC}"
    echo "  bdf:    Device address (e.g., 01:00.0)"
    echo "  action: 'gen4' or 'reset'"
    exit 1
}

# --- Script Main Logic ---

# 1. Input Validation and Sanity Checks
# =======================================

# Check for root privileges
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root.${NC}"
   exit 1
fi

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    print_usage
fi

BDF="$1"
ACTION="$2"

# Validate BDF format (simple regex)
if ! [[ "$BDF" =~ ^[0-9a-fA-F]{2,4}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$ ]]; then
    echo -e "${RED}Error: Invalid BDF format. Expected format is bb:dd.f (e.g., 01:00.0).${NC}"
    exit 1
fi

# Check if the device exists
if ! lspci -s "$BDF" > /dev/null; then
    echo -e "${RED}Error: Device with BDF '$BDF' not found.${NC}"
    exit 1
fi

# 2. Find PCIe Capability Offset
# =======================================

echo "INFO: Locating PCI Express capability for device $BDF..."
# Use lspci to find the capability offset. The output looks like:
#   Capabilities: [40] Express (v2) Root Port (Slot+), MSI 00
CAP_OFFSET_HEX=$(lspci -s "$BDF" -vv | grep 'Capabilities:.*Express' | head -n 1 | awk -F'[][]' '{print $2}')

if [ -z "$CAP_OFFSET_HEX" ]; then
    echo -e "${RED}Error: Could not find PCI Express capability structure for device $BDF.${NC}"
    echo "       Please ensure this is a valid PCIe device."
    exit 1
fi

# Convert hex offset to decimal for calculations
CAP_OFFSET_DEC=$((16#$CAP_OFFSET_HEX))
echo -e "${GREEN}Success: Found Express capability at offset 0x${CAP_OFFSET_HEX}.${NC}"

# 3. Determine Target Speed
# =======================================

TARGET_SPEED_VAL=0

case "$ACTION" in
    gen4)
        TARGET_SPEED_VAL=4
        echo "INFO: Action is 'gen4'. Target speed set to 16.0 GT/s."
        ;;
    reset)
        echo "INFO: Action is 'reset'. Determining max supported speed..."
        # Calculate address of the Link Capabilities Register
        LINK_CAP_REG_ADDR=$((CAP_OFFSET_DEC + LINK_CAP_REG_OFFSET))
        # Read the 32-bit register value
        LINK_CAP_VAL=$(setpci -s "$BDF" "$(printf '%x' $LINK_CAP_REG_ADDR).L")
        # The max speed is in the lowest 4 bits (mask with 0xf)
        TARGET_SPEED_VAL=$((16#$LINK_CAP_VAL & 0xf))
        echo -e "${GREEN}Success: Device reports max speed as Gen${TARGET_SPEED_VAL}.${NC}"
        ;;
    *)
        echo -e "${RED}Error: Invalid action '$ACTION'.${NC}"
        print_usage
        ;;
esac

# 4. Apply Speed Change and Retrain Link
# =======================================

# Calculate final addresses for the control registers
LINK_CTRL_2_ADDR_HEX=$(printf '%x' $((CAP_OFFSET_DEC + LINK_CTRL_2_REG_OFFSET)))
LINK_CTRL_ADDR_HEX=$(printf '%x' $((CAP_OFFSET_DEC + LINK_CTRL_REG_OFFSET)))

echo "INFO: Writing target speed 'Gen${TARGET_SPEED_VAL}' to Link Control 2 Register (at 0x${LINK_CTRL_2_ADDR_HEX})."
# Write the target speed value, masking to only change the lowest 4 bits
setpci -s "$BDF" "${LINK_CTRL_2_ADDR_HEX}.W=${TARGET_SPEED_VAL}:f"

echo "INFO: Retraining the PCIe link..."
# Read the current value of the Link Control Register
CURRENT_LINK_CTRL_VAL=$(setpci -s "$BDF" "${LINK_CTRL_ADDR_HEX}.W")
# Set bit 5 (0x20) to trigger a link retrain
NEW_LINK_CTRL_VAL=$(printf '%x' $((0x$CURRENT_LINK_CTRL_VAL | 0x20)))
# Write the new value back to the register
setpci -s "$BDF" "${LINK_CTRL_ADDR_HEX}.W=${NEW_LINK_CTRL_VAL}"

# 5. Verification
# =======================================

echo "INFO: Waiting a moment for the link to retrain..."
sleep 1 # Give the hardware a second to renegotiate

echo -e "${GREEN}Configuration complete. Verifying current link status...${NC}"
# Display the LnkSta field, which shows the negotiated speed and width
lspci -s "$BDF" -vv | grep "LnkSta:"
