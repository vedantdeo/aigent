#!/bin/sh
# Print the cltrplusf commit the uv.lock on stdin pins.
sed -n "s|^source = { git = \"$CLTRPLUSF?[^#]*#\([0-9a-f]*\)\" }\$|\1|p"
