#!/bin/bash

# Disable timeouts for long-running jobs on notebooks
uv run --with jupyter jupyter lab --ip=0.0.0.0 --port=8888 --ServerApp.shutdown_no_activity_timeout=0 --MappingKernelManager.cull_idle_timeout=0