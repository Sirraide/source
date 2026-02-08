#!/usr/bin/env bash
for i in test/CG/*.src; do
    ./update-test.sh "$i" &
done

wait
echo "DONE! All CodeGen tests updated."