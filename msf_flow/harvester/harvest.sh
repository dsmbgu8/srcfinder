#!/bin/sh
#
# Prerequisite: Set environment variable CEDAS_HOME to point to the base 
# directory containing the src directory containing the msf_flow repo and 
# the Anaconda distribution for python3.
MSF_FLOW_HOME=${CEDAS_HOME}/src/msf_flow
PY_EXEC=${CEDAS_HOME}/anaconda3/bin/python
PY_SCRIPT=${MSF_FLOW_HOME}/harvester/harvest.py
export PYTHONPATH=${MSF_FLOW_HOME}:${PYTHONPATH}
echo `date` Calling ${PY_EXEC} ${PY_SCRIPT} "$@"
exec ${PY_EXEC} ${PY_SCRIPT} "$@"
echo `date` Finished ====================================
