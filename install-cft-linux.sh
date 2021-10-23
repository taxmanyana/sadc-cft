#!/bin/bash
#
# This script downloads and compiles source code of dependencies to install the CFT
#
echo "installing SADC CFT..."
echo
SCRIPT=$(readlink -f $0)
cwd=`dirname $SCRIPT`
echo "CFT directory: $cwd"
echo
export PYTHONPATH=$cwd/python3
export PATH=${PYTHONPATH}/bin:$PATH 
export LD_LIBRARY_PATH=${PYTHONPATH}/lib:${PYTHONPATH}/lib64:$LD_LIBRARY_PATH 
export LDFLAGS="-L${PYTHONPATH}/lib -L${PYTHONPATH}/lib64" 
export LD_RUN_PATH=${PYTHONPATH}/lib:${PYTHONPATH}/lib64
export CPPFLAGS="-I${PYTHONPATH}/include" 

cd $cwd
mkdir -p source 
if [ -d source ]; then
  cd $cwd/source
  wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.bz2 -c
  lib=$(find ./ -maxdepth 1 -name "openmpi-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing ${lib##*/}..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "openmpi-*" -type d)
    ./configure --prefix=${PYTHONPATH}
    make install -j4
  fi
  cd $cwd/source
  wget https://www.openssl.org/source/openssl-3.0.0-alpha6.tar.gz -c
  lib=$(find ./ -maxdepth 1 -name "openssl-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing ${lib##*/}..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "openssl-*" -type d)
    ./config --prefix=${PYTHONPATH} --openssldir=${PYTHONPATH}
    make -j4
    make install -j4
  fi
  cd $cwd/source
  wget https://gcc.gnu.org/pub/libffi/libffi-3.3.tar.gz -c
  lib=$(find ./ -maxdepth 1 -name "libffi-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing libffi..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "libffi-*" -type d)
    ./configure --prefix=${PYTHONPATH}
    make install -j4
  fi
  cd $cwd/source
  wget https://deac-fra.dl.sourceforge.net/project/lzmautils/xz-5.2.5.tar.xz -c
  lib=$(find ./ -maxdepth 1 -name "xz-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing xz..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "xz-*" -type d)
    ./configure --prefix=${PYTHONPATH}
    make install -j4
  fi
  cd $cwd/source
  wget https://zlib.net/zlib-1.2.11.tar.gz -c
  lib=$(find ./ -maxdepth 1 -name "zlib-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing zlib..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "zlib-*" -type d)
    ./configure --prefix=${PYTHONPATH}
    make install -j4
  fi
  cd $cwd/source
  wget https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz --no-check-certificate -c
  lib=$(find ./ -maxdepth 1 -name "bzip2-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing bzip2..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "bzip2-*" -type d)
    make
    make clean
    make -f Makefile-libbz2_so
    makelink=$(grep "ln -s " Makefile-libbz2_so | xargs)
    libbz2_so=$(echo $makelink | awk '{print $3}')
    make install PREFIX=${PYTHONPATH}
    cp -v $libbz2_so ${PYTHONPATH}/lib
    cd ${PYTHONPATH}/lib
    $makelink
  fi
  cd $cwd/source
  wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tar.xz -c
  lib=$(find ./ -maxdepth 1 -name "Python-*" -type f | tail -1)
  if [ -n $lib ]; then
    echo
    echo "installing python..."
    tar xf $lib
    cd $(find ./ -maxdepth 1 -name "Python-*" -type d)
    ./configure --prefix=${PYTHONPATH} --with-openssl=${PYTHONPATH} --enable-optimizations
    make -j4
    make install
  fi
fi

echo "installing python libraries..."
cd $cwd
echo "upgrading pip"
pip3 install --upgrade pip
echo
echo "installing other dependencies..."
if [ ! -f $cwd/requirements.txt ]; then
   echo "could not locate requirements file. exit"
   exit 1
fi 
pip3 install --upgrade -r $cwd/requirements.txt
if [ $? -ne 0 ]; then
  exit
fi
if [ -n $(which mpirun) ]; then
  pip3 install --upgrade  mpi4py
  [ $? -ne 0 ] && echo "MPI support installed"
fi
echo
echo "creating environmental variables file"
echo "export PATH=${PYTHONPATH}/bin:$PATH" > ${PYTHONPATH}/bin/activate
echo "export LD_LIBRARY_PATH=${PYTHONPATH}/lib:$LD_LIBRARY_PATH" >> ${PYTHONPATH}/bin/activate
echo "export LD_RUN_PATH=${PYTHONPATH}/lib:${PYTHONPATH}/lib64" >> ${PYTHONPATH}/bin/activate
echo
echo "creating desktop shortcut"
echo "#!/usr/bin/env xdg-open" > $cwd/CFT.desktop
echo "[Desktop Entry]" >> $cwd/CFT.desktop
echo "Version=1.0" >> $cwd/CFT.desktop
echo "Type=Application" >> $cwd/CFT.desktop
echo "Terminal=false" >> $cwd/CFT.desktop
echo "Exec=${cwd}/cft_ubuntu.sh" >> $cwd/CFT.desktop
echo "Name=CFT" >> $cwd/CFT.desktop
echo "Comment=CFT" >> $cwd/CFT.desktop
echo "Icon=${cwd}/sadc.ico" >> $cwd/CFT.desktop
chmod 777 $cwd/CFT.desktop
chmod +x ${cwd}/cft_ubuntu.sh
[ -d ~/Desktop ]  && cp $cwd/CFT.desktop ~/Desktop/

python3 <<END
from multiprocessing import Pool, cpu_count
from netCDF4 import Dataset
import geojson, json
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from scipy.stats import pearsonr
from shapely.geometry import shape, Point
from sklearn import linear_model, cluster
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
import statsmodels.api as sm
END
if [ $? -eq 0 ]; then
   echo "installation successfully completed"
   rm -fr source
else
   echo "something went wrong during the installation :-("
fi
