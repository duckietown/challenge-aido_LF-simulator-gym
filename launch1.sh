#!/bin/bash

prefix="$0 > "
_kill_procs() {
   echo "$prefix _kill_procs()"
#  kill -TERM $gym
#  wait $gym
  kill -TERM $xvfb
}


## Setup a trap to catch SIGTERM and relay it to child processes
trap _kill_procs SIGTERM
trap _kill_procs INT

# make sure there is no xserver running (only neded for docker-compose)
#killall Xvfb || true

echo "Listing /tmp"
ls -a /tmp
rm -f /tmp/.X99-lock || true

# Start Xvfbs
echo "$prefix Starting xvfb"
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
xvfb=$!
echo "$prefix Started xvfb with PID $xvfb" 1>&2
export DISPLAY=:99

sleep 2

echo "$prefix Now running [$@]" 1>&2
nice -n 20 $@
ret=$?
echo "$prefix command [$@] terminated with return code $ret" 1>&2


echo "$prefix Killing xvfb..." 1>&2
kill $xvfb

echo "$prefix Killed." 1>&2

echo "$prefix Graceful exit of launch.sh with return code $ret" 1>&2
exit $ret
