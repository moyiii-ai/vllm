#!/bin/bash
echo -e "PID\tNode0_MB\tNode1_MB\tTotal_MB\tCMD"

for pid in $(ls /proc | grep '^[0-9]'); do
    if [ -r /proc/$pid/cmdline ]; then
        out=$(numastat -p $pid 2>/dev/null | awk '/^Total/ {print $2, $3, $4}')
        if [ -n "$out" ]; then
            node0=$(echo $out | awk '{print int($1)}')
            node1=$(echo $out | awk '{print int($2)}')
            total=$(echo $out | awk '{print int($3)}')
            if [ "$total" -gt 100 ]; then
                cmd=$(tr -d '\0' < /proc/$pid/cmdline | cut -c1-50)
                echo -e "$pid\t$node0\t$node1\t$total\t$cmd"
            fi
        fi
    fi
done | sort -k3 -nr | head -20

numastat

cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|Slab|Huge|Buffers|Cached'

for pid in $(ls /proc | grep '^[0-9]'); do numastat -p $pid 2>/dev/null | awk -v p=$pid '/Total/ {print p,$3}' ; done | sort -nr | head

