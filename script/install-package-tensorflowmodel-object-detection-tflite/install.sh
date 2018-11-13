#! /bin/bash

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
mkdir tmp
cd tmp
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
unzip ${PACKAGE_NAME}
mv ${FLATBUFFER} ..
mv ${LABELMAP_FILE} ..
mv ${ANNOTATIONS} ..

########################################################################
echo
echo "Remove temporary files ..."
cd ..
rm -rf tmp

#####################################################################
#echo ""
#echo "Copy label-map file"
#cp -f ${CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR}/data/${LABELMAP_FILE} .

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' tensorflow model ..."
exit 0
