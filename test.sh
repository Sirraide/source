#!/usr/bin/env bash
ctest --test-dir out -j`nproc` --progress --output-on-failure
