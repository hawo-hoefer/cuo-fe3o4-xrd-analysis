#!/bin/bash

CLEAN=0
ONLY_REF=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=1
      shift
      ;;
    --only-ref)
      ONLY_REF=1
      shift
      ;;
    *)
      echo "Invalid Argument '$1'."
      echo "usage: ./generate.sh [opt]"
      echo "options:" 
      echo "    --clean      remove generated data"
      echo "    --only-ref   only generate reference pattern"
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

ln -srf ./cif/ ./data/ref_pattern/cif
pushd ./data/ref_pattern/
if [[ $CLEAN -eq 0 ]]; then
  yaxs ./ref.yaml -o ref --overwrite
else
  rm ./ref -rf
fi
popd

if [[ $ONLY_REF -eq 1 ]]; then
  exit
fi

dirs=(
  "cuka"
  # "cukab"
)

for dir in ${dirs[@]}; do
  ln -srf ./cif/ ./exp_data_analysis/$dir/cif
  pushd ./exp_data_analysis/$dir
    if [[ $CLEAN -eq 0 ]]; then
      yaxs "./train.yaml" -o "train" -c 160000 --overwrite
      yaxs "./val.yaml" -o "val" -c 160000 --overwrite
    else
      rm "./train/" -rf
      rm "./val/" -rf
    fi
  popd
done
