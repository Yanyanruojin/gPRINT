#!/bin/bash
echo "Reference_data is : $1"
echo "Query_data is : $2"

Rscript gprint_group_pre.R $1 $2
echo "The reordered data has been saved."

python gprint_group.py $1 $2
echo "Cell annotation is complete."

