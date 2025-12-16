#!/bin/bash

CLEAN=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=1
      shift
      ;;
    *)
      echo "Invalid Argument '$1'. Expected '--clean' to remove data or nothing"
      exit 1
      ;;
  esac
done

if [[ $CLEAN -eq 0 ]]; then
  echo "generating data"
else
  echo "cleaning up data"
fi

set -xeo pipefail

dirs=(
  "c3"
  "c4"
  "cuka"
  "cuka1"
  "cukab"
)

for dir in ${dirs[@]}; do
  ln -srf ./cif/ ./data/$dir/cif
  pushd ./data/$dir
    if [[ $CLEAN -eq 0 ]]; then
      yaxs ./data-train.yaml -o train -c 160000 --overwrite
      yaxs ./data-val.yaml -o val -c 160000 --overwrite
    else
      rm "./train/" -rf
      rm "./val/" -rf
      rm "./test/" -rf
    fi
  popd
done
