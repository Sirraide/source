#!/usr/bin/env bash

set -eu

./fchk/out/fchk "$1" --prefix // -P nocap -P captype -D "srcc=./srcc" --update

## Remove metadata nodes.
sed -E -i '\%//[^\s]*\s+[+*]\s+!%d' "$1"
