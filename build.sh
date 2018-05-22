#! /bin/bash
set -e

export TC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if ! test ${CLANG_PREFIX}; then
    echo 'Environment variable CLANG_PREFIX is required, please run export CLANG_PREFIX=$(llvm-config --prefix)'
    exit 1
fi

WITH_CAFFE2=${WITH_CAFFE2:=ON}
WITH_CUDA=${WITH_CUDA:=ON}
if [ "${WITH_CUDA,,}" = "off" -o "${WITH_CUDA,,}" = "no" -o "${WITH_CUDA}" = "0" ]; then
  ATEN_NO_CUDA=1
else
  ATEN_NO_CUDA=${ATEN_NO_CUDA:=0}
fi
WITH_NNPACK=${WITH_NNPACK:=OFF}
WITH_TAPIR=${WITH_TAPIR:=ON}
PYTHON=${PYTHON:="`which python3`"}
PROTOC=${PROTOC:="`which protoc`"}
CORES=${CORES:=32}
VERBOSE=${VERBOSE:=0}
CMAKE_VERSION=${CMAKE_VERSION:="`which cmake3 || which cmake`"}
HALIDE_BUILD_CACHE=${HALIDE_BUILD_CACHE:=${TC_DIR}/third-party/.halide_build_cache}
INSTALL_PREFIX=${INSTALL_PREFIX:=${TC_DIR}/third-party-install/}
CCACHE_WRAPPER_DIR=${CCACHE_WRAPPER_DIR:=/usr/local/bin/}
CC=${CC:="`which gcc`"}
CXX=${CXX:="`which g++`"}

echo $TC_DIR $GCC_VER
BUILD_TYPE=${BUILD_TYPE:=Debug}
echo "Build Type: ${BUILD_TYPE}"

clean=""
if [[ $* == *--clean* ]]
then
    echo "Forcing clean"
    clean="1"
fi

dlpack=""
if [[ $* == *--dlpack* ]]
then
    echo "Building DLPACK"
    dlpack="1"
fi

tc=""
if [[ $* == *--tc* ]]
then
    echo "Building TC"
    tc="1"
fi

halide=""
if [[ $* == *--halide* ]]
then
    echo "Building Halide"
    halide="1"
fi

all=""
if [[ $* == *--all* ]]
then
    echo "Building ALL"
    all="1"
fi

if [[ "${VERBOSE}" = "1" ]]; then
  set -x
fi

orig_make=$(which make)

function make() {
    # Workaround for https://cmake.org/Bug/view.php?id=3378
    if [[ "${VERBOSE}" = "1" ]]; then
        VERBOSE=${VERBOSE} ${orig_make} $@
    else
        ${orig_make} $@
    fi
}

function set_cache() {
    stat --format="%n %Y %Z %s" `find $1 -name CMakeLists.txt -o -name autogen.sh -o -name configure -o -name Makefile -exec realpath {} \;` > $2
}

function should_reconfigure() {
    if [ "$clean" = "1" ]
    then
        true
    else
        if [ -e $2 ]
        then
            OLD_STAT=`cat $2`
            NEW_STAT=$(stat --format="%n %Y %Z %s" `find $1 -name CMakeLists.txt -o -name autogen.sh -o -name configure -o -name Makefile -exec realpath {} \;`)
            if [ "$OLD_STAT" = "$NEW_STAT" ]
            then
                false
            else
                true
            fi
        else
            true
        fi
    fi
}

function set_bcache() {
    stat --format="%n %Y %Z %s" $1 > $2
}

function should_rebuild() {
    if [ "$clean" = "1" ]
    then
        true
    else
        if [ -e $2 ]
        then
            OLD_STAT=`cat $2`
            NEW_STAT=$(stat --format="%n %Y %Z %s" $1)
            if [ "$OLD_STAT" = "$NEW_STAT" ]
            then
                false
            else
                true
            fi
        else
            true
        fi
    fi
}

function install_dlpack() {
  mkdir -p ${TC_DIR}/third-party/dlpack/build || exit 1
  cd       ${TC_DIR}/third-party/dlpack/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -d "${INSTALL_PREFIX}/include/dlpack" ]; then

  if should_rebuild ${TC_DIR}/third-party/dlpack ${TC_DIR}/third-party/.dlpack_build_cache; then
  echo "Installing DLPACK"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring DLPACK"
    rm -rf * || exit 1
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
  fi
  make -j $CORES -s || exit 1

  set_cache .. .build_cache
  set_bcache ${TC_DIR}/third-party/dlpack ${TC_DIR}/third-party/.dlpack_build_cache
  fi

  # make install -j $CORES -s || exit 1
  cp -R ${TC_DIR}/third-party/dlpack/include/dlpack ${INSTALL_PREFIX}/include/
  echo "Successfully installed DLPACK"

  fi
}

function install_cub() {
  local tp_dir=${TC_DIR}/third-party/cub/cub
  local include_dir=${INSTALL_PREFIX}/include/
  if diff -rq ${tp_dir} ${include_dir}/cub >/dev/null 2>&1; then
    echo "CUB is up to date"
  else
    echo "Installing CUB"
    cp -R ${tp_dir} ${include_dir}
  fi
}

function install_tc_python() {
    echo "Setting up python now"
    echo "USE_CONTBUILD_CACHE: ${USE_CONTBUILD_CACHE}"

    if [ "$USE_CONTBUILD_CACHE" == "1" ]; then
      echo "Running on CI, setting PYTHONPATH only"
      export PYTHONPATH=${TC_DIR}/build/tensor_comprehensions/pybinds:${PYTHONPATH}
      echo "PYTHONPATH: ${PYTHONPATH}"
    else
      if [[ $(conda --version | wc -c) -ne 0 ]]; then
          echo "Found conda, going to install Python packages"
          conda install -y mkl-include
          cd ${TC_DIR}
          export CONDA_PYTHON=$(which python3)
          echo "CONDA_PYTHON: ${CONDA_PYTHON}"
          if [ "$BUILD_TYPE" == "Release" ]; then
            echo "Install mode setup for python"
            ${CONDA_PYTHON} setup.py install
          else
            echo "Develop mode setup for python"
            ${CONDA_PYTHON} setup.py develop
          fi
      else
          echo "Conda not found, setting PYTHONPATH instead"
          echo "Setting PYTHONPATH now"
          export PYTHONPATH=${TC_DIR}/tensor_comprehensions:$PYTHONPATH
          echo "PYTHONPATH: ${PYTHONPATH}"
      fi
    fi
    echo "python all set now"
}

function install_tc() {
  install_cub

  mkdir -p ${TC_DIR}/build || exit 1
  cd       ${TC_DIR}/build || exit 1

  echo "Installing TC"

  if should_reconfigure .. .build_cache; then
    echo "Reconfiguring TC"
    rm -rf *
    VERBOSE=${VERBOSE} ${CMAKE_VERSION} -DWITH_CAFFE2=${WITH_CAFFE2} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_TAPIR=${WITH_TAPIR} \
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}/lib/cmake \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DPROTOBUF_PROTOC_EXECUTABLE=${PROTOC} \
        -DCLANG_PREFIX=${CLANG_PREFIX} \
        -DCUDNN_ROOT_DIR=${CUDNN_ROOT_DIR} \
        -DWITH_CUDA=${WITH_CUDA} \
        -DTC_DIR=${TC_DIR} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} .. || exit 1
  fi

  set_cache .. .build_cache
  make -j $CORES -s || exit 1
  make install -j $CORES -s || exit 1

  install_tc_python

  echo "Successfully installed TC"
}

function install_halide() {
  mkdir -p ${TC_DIR}/third-party/halide/build || exit 1
  cd       ${TC_DIR}/third-party/halide/build || exit 1

  if ! test ${USE_CONTBUILD_CACHE} || [ ! -e "${INSTALL_PREFIX}/include/Halide.h" ]; then
    LLVM_CONFIG_FROM_PREFIX=${CLANG_PREFIX}/bin/llvm-config
    LLVM_CONFIG=$( which $LLVM_CONFIG_FROM_PREFIX || which llvm-config-4.0 || which llvm-config )
    CLANG_FROM_PREFIX=${CLANG_PREFIX}/bin/clang
    CLANG=$( which $CLANG_FROM_PREFIX || which clang-4.0 || which clang )

    if should_rebuild ${TC_DIR}/third-party/halide ${HALIDE_BUILD_CACHE}; then
      CLANG=${CLANG} \
      LLVM_CONFIG=${LLVM_CONFIG} \
      VERBOSE=${VERBOSE} \
      PREFIX=${INSTALL_PREFIX} \
      WITH_LLVM_INSIDE_SHARED_LIBHALIDE= \
      WITH_OPENCL= \
      WITH_OPENGL= \
      WITH_METAL= \
      WITH_EXCEPTIONS=1 \
      make -f ../Makefile -j $CORES || exit 1
      set_bcache ${TC_DIR}/third-party/halide ${HALIDE_BUILD_CACHE}
    fi

    CLANG=${CLANG} \
    LLVM_CONFIG=${LLVM_CONFIG} \
    VERBOSE=${VERBOSE} \
    PREFIX=${INSTALL_PREFIX} \
    WITH_LLVM_INSIDE_SHARED_LIBHALIDE= \
    WITH_OPENCL= \
    WITH_OPENGL= \
    WITH_METAL= \
    WITH_EXCEPTIONS=1 \
    make -f ../Makefile -j $CORES install || exit 1

    echo "Successfully installed Halide"

  fi
}

if ! test -z $dlpack || ! test -z $all; then
    install_dlpack
fi

if ! test -z $halide || ! test -z $all; then
    if [[ ! -z "$CONDA_PREFIX" && $(find $CONDA_PREFIX -name libHalide.so) ]]; then
        echo "Halide found"
    else
        echo "no files found"
        install_halide
    fi
fi

if ! test -z $tc || ! test -z $all; then
    install_tc
fi
