#!/bin/bash
uv run mcp dev minimax_mcp/server.py --with python-dotenv --with fuzzywuzzy --with python-Levenshtein --with sounddevice --with soundfile --with-editable .
