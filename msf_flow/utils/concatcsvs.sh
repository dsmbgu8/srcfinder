#!/bin/bash
# Concatinates a folder of CSVs into a single output CSV.
# Preserves the header (1st line) of the first CSV processed.
# To Run:
# $ concatcsvs.sh <input directory> <output csv file>

indir=$1
suffix=$2
outfile=$3

n=0
for f in `ls $indir/*$suffix`; do
  if [ $n -eq 0 ]; then
    cat $f > $outfile
  else
    tail -n +2 $f >> $outfile
  fi
  let n=1
done
