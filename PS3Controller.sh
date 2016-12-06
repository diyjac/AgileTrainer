#!/bin/sh
# This shell script sets up a PS3 controller
xboxdrv --detach-kernel-driver --led 2 --silent &
#jstest-gtk
#js-store /dev/input/js0

