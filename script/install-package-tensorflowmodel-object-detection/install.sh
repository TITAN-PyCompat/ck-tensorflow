#! /bin/bash
#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
mkdir tmp
cd tmp
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."

if [ "${PACKAGE_UNZIP}" == "YES" ]; then
  unzip ${PACKAGE_NAME}
elif [ "${PACKAGE_UNTARGZ}" == "YES" ]; then
  tar -zxvf ${PACKAGE_NAME}
else
  echo
  echo 'ERROR: Unknown how to unpack downloaded package'
  exit -1
fi

########################################################################
echo
echo "Copy weights files ..."

if [ ! -z "${FROZEN_GRAPH}" ]; then
  mv ${PACKAGE_NAME1}/${FROZEN_GRAPH} ..
fi
if [ ! -z "${WEIGHTS_FILE}" ]; then
  mv ${PACKAGE_NAME1}/${WEIGHTS_FILE}* ..
fi
if [ ! -z "${TFLITE_FILE}" ]; then
  mv ${PACKAGE_NAME1}/${TFLITE_FILE} ..
fi

########################################################################
echo
echo "Remove temporary files ..."
cd ..
rm -rf tmp

#####################################################################
echo ""
echo "Copy label-map file"
cp -f ${CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR}/data/${LABELMAP_FILE} .

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' tensorflow model ..."
exit 0
