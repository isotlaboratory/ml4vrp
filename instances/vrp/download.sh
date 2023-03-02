#!/usr/bin/env bash
set -euo pipefail

script_path=$(dirname "$0")

# Reading, downloading and extracting files from the url
while read url; do
    wget ${url} -P ${script_path}
    unzip -o ${script_path}/$(sed 's#.*/##' <<< ${url}) -d ${script_path}
done < ${script_path}/urls.txt
