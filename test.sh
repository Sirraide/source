#!/usr/bin/env bash
set -eu
ctest --test-dir out/ -j`nproc` --progress --output-on-failure --max-width 200 --timeout 5s
