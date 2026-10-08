#!/bin/sh
# Print the underhood commit the uv.lock on stdin pins.
sed -n "s|^source = { git = \"$UNDERHOOD?[^#]*#\([0-9a-f]*\)\" }\$|\1|p"
