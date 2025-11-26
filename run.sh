#!/usr/bin/env bash
set -e


PIPE="nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1,format=NV12 \
 ! nvvidconv ! video/x-raw,width=416,height=416,format=BGRx ! videoconvert ! video/x-raw,format=BGR \
 ! appsink drop=1 max-buffers=1"


SHOW=0
if [[ "$1" == "--show" ]]; then
  SHOW=1
fi

cd "$(dirname "$0")/build"

if [[ $SHOW -eq 1 ]]; then
  ./avdet "$PIPE"
else
  ./avdet "$PIPE" --no-display
fi
