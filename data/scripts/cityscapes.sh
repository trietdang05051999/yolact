#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ./data
  else
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit 0
    fi
    echo "navigating to " $1 " ..."
    cd $1
fi

if [ ! -d images ]
  then
    echo "There's no images folder"
fi

cd ../
if [ ! -d annotations ]
  then
    echo "There's no annotations folder"
fi


end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"