#!/bin/bash

PYTHON_VERSION="2 and 3"

function p2 {
  apt-get install python-pip
  python2 -m pip install --upgrade pip
  python2 -m pip install jupyter numpy pandas matplotlib pillow
}

function p3 {
  apt-get install python3-pip
  python3 -m pip install --upgrade pip
  python3 -m pip install jupyter numpy pandas matplotlib pillow
}

while [[ $# -gt 1 ]]
do
  key="$1"

  case $key in
    --python_version)
    PYTHON_VERSION="$2"
  shift # past argument
    ;;
    *)
    # unknown option
    ;;
  esac

  shift # past argument or value
done

echo PYTHON_VERSION  = "${PYTHON_VERSION}"

if [ "${PYTHON_VERSION}" == "2 and 3" ]; then
  p2
  p3
elif [ "${PYTHON_VERSION}" == "2" ]; then
 p2
elif [ "${PYTHON_VERSION}" == "3" ]; then
 p3
else
  echo 'bad argument'
fi
