#!/bin/bash
echo "test is ready"
echo "Query_data is : $1"

cd .. 
cd code/

python test.py
echo "test is ok"
