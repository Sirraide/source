#!/usr/bin/env bash

set -eu

./fchk/out/fchk "$1" --prefix // -P nocap -P captype -D "srcc=./srcc" --update

## Remove '!{!"clang version' metadata node.
sed -i '/!{!"clang version/d' "$1"
