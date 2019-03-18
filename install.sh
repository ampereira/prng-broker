#!/bin/bash
#
# Script that installs the PRNGM library
# 

# export SUFFIX="scripts"
# export HEP_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# export HEP_DIR=${HEP_DIR%$SUFFIX}

source scripts/realpath.sh


if [ "$#" -eq 0 ]
  then
    echo "Insufficient parameters. Sample execution:"
    echo "./install.sh /boost/directory/ /directory/to/install/PRNGM"
    echo "Note that the second parameter is optional."
    echo "If no directory is set the library will be compiled in the current path."
    exit
fi

if [ "$#" -gt 0 ]
  then
    export BOOST_DIR=$(realpath $1)
fi

mv Makefile Makefile_temp
line=$(head -n 1 Makefile_temp)
if [[ "$line" == BOOST* ]]; then
  echo "$(tail -n +2 Makefile_temp)" > Makefile_temp
fi
echo "BOOST_DIR=${BOOST_DIR}" | cat - Makefile_temp > Makefile
rm Makefile_temp

make clean
make

if [ "$#" -eq 2 ]
  then
    export PREFIX=$(realpath $2)
    
    mkdir -p $PREFIX/include
    mkdir -p $PREFIX/lib

    cp src/*.h $PREFIX/include/
    cp lib/* $PREFIX/lib/
fi

