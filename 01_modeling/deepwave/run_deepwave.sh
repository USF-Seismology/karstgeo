#!/bin/bash
if [ "$CONDA_DEFAULT_ENV" != "pytorchenv" ]; then
    conda activate pytorchenv
fi
python ./deepwave_karst_model_v2.py --config ./karst_survey_config_v3.yml
