#!/bin/bash

REPO_PATH="${HOME}/repos/iros-2021-sdsmm"
DIR_FOR_ENV="${REPO_PATH}/scripts/env"
ENV_NAME="iros21_sdsmm"
ENV_PATH="${DIR_FOR_ENV}/${ENV_NAME}"

pushd "${DIR_FOR_ENV}"

# Check if virtualenv is installed.
if [[ ! $(command python3 -m venv -h) ]]; then
    printf "Cannot find virtualenv.\n"
fi

# Check if the environment was set up.
EXISTS=0

if [[ ! -d "${ENV_PATH}" ]]; then
    printf "The $ENV_NAME environment does not exist.\n"

    RESPONSE=""
    while [[ "$RESPONSE" != "y" && "$RESPONSE" != "n" ]]; do
        printf "Would you like me to create the environment? (y/n) "
        read RESPONSE
        printf "\n"
    done

    if [[ "$RESPONSE" == "y" ]]; then
        printf "Creating a virtual environment for $ENV_NAME.\n"
        python3 -m venv ${ENV_NAME} && EXISTS=1
        printf "Once the environment is activated, run ''bash install_packages.sh''\n"
    fi
else
    EXISTS=1
fi

# Activate the environment if it exists.
if [[ $EXISTS == 0 ]]; then
    printf "Cannot activate $ENV_NAME because it does not exist.\n"

else
    export IROS21_SDSMM="${REPO_PATH}"
    export PYTHONPATH=$IROS21_SDSMM:$PYTHONPATH
    source ${ENV_NAME}/bin/activate
fi

popd
