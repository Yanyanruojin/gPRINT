#!/bin/bash
echo "Reference_data is : $1"
echo "Query_data is : $2"
cd .. 
cd code/

Rscript gprint_pre.R $1 $2
echo "The reordered data has been saved."

python gprint.py $1 $2
echo "Cell annotation is complete."
