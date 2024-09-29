#!/bin/bash
echo "Query_data is : $1"
echo "Reference_model is : $2"

cd .. 
cd code/

conda activate Rstudio
Rscript gprint_pre_reference.R $1 $2
echo "The reordered data has been saved."

conda activate Keras
python gprint_reference.py $1 $2
echo "Cell annotation is complete."


