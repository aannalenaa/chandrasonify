#!/bin/bash

source /Applications/ciao-4.18/bin/ciao.sh

# this should output something like:
# CIAO 4.18.0 Monday, December 08, 2025
#   bindir      : /Applications/ciao-4.18/bin
#   CALDB       : 4.12.3


obs_ids=(22304)  # enter your observations, this can be a list


# below uncomment the version you want to run and comment out the others
for id in "${obs_ids[@]}"; do
    echo "Processing OBS_ID: $id"
    # python -m chandrasonify.manual_code -o "$id"
    # python -m chandrasonify.llm_code -o "$id"
    python -m chandrasonify.agentic_code_base -o "$id"
done