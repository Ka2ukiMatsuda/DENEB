#!/bin/bash
alias python_='python'

yecho(){ # yellow echo
    echo "\e[1;33m$1\e[m"
}

arg=$1 
if [ "$arg" = "pdb" ]; then
    yecho 'Launch pdb ...'
    alias python_='python -m pdb -c c'
else 
    yecho 'Launch without pdb ...'
fi

python_ deneb/cli.py train -f configs/xlmr/base/da-estimator-cvpr.yaml && \
sh validate.sh
