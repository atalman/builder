if [[ ${MATRIX_PACKAGE_TYPE} == "libtorch" ]]; then
    curl ${MATRIX_INSTALLATION} -o libtorch.zip
    unzip libtorch.zip
else

    conda update -y -n base -c defaults conda
    # Please note ffmpeg is required for torchaudio, see https://github.com/pytorch/pytorch/issues/96159
    conda create -y -n ${ENV_NAME} python=${MATRIX_PYTHON_VERSION} numpy ffmpeg
    conda activate ${ENV_NAME}
    INSTALLATION=${MATRIX_INSTALLATION/"conda install"/"conda install -y"}
    TEST_SUFFIX=""
    if [[ ${TORCH_ONLY} == 'true' ]]; then
        INSTALLATION=${INSTALLATION/"torchvision torchaudio"/""}
        TEST_SUFFIX=" --package torchonly"
    fi

    # if RELESE version is passed as parameter - install speific version
    if [[ ! -z ${RELEASE_VERSION} ]]; then
          INSTALLATION=${INSTALLATION/"torch "/"torch==${RELEASE_VERSION} "}
          INSTALLATION=${INSTALLATION/"-y pytorch "/"-y pytorch==${RELEASE_VERSION} "}
          INSTALLATION=${INSTALLATION/"::pytorch "/"::pytorch==${RELEASE_VERSION} "}
    fi

    export OLD_PATH=${PATH}
    # Workaround macos-arm64 runners. Issue: https://github.com/pytorch/test-infra/issues/4342
    if [[ ${TARGET_OS} == 'macos-arm64' ]]; then
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
    fi

    # Make sure we remove previous installation if it exists, this issue seems to affect only
    if [[ ${MATRIX_PACKAGE_TYPE} == 'wheel' ]]; then
        pip3 uninstall -y torch torchaudio torchvision
    fi
    eval $INSTALLATION

    if [[ ${TARGET_OS} == 'linux' ]]; then
        export CONDA_LIBRARY_PATH="$(dirname $(which python))/../lib"
        export LD_LIBRARY_PATH=$CONDA_LIBRARY_PATH:$LD_LIBRARY_PATH
        ${PWD}/check_binary.sh
    fi

    if [[ ${TARGET_OS} == 'windows' ]]; then
        python  ./test/smoke_test/smoke_test.py ${TEST_SUFFIX}
    else
        python3  ./test/smoke_test/smoke_test.py ${TEST_SUFFIX}
    fi

    if [[ ${TARGET_OS} == 'macos-arm64' ]]; then
        export PATH=${OLD_PATH}
    fi

    if [[ ${INCLUDE_TEST_OPS} == 'true' ]]; then
        source ./.github/scripts/validate_test_ops.sh
    fi

    conda deactivate
    conda env remove -n ${ENV_NAME}
fi
