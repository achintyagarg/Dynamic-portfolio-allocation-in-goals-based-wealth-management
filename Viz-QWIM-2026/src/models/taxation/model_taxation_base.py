"""
Annuity base class
===============================

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

import logging
import os
import sys
import typing
import platform
from datetime import datetime
from pathlib import Path
import asyncio
import traceback

import numpy as np
import pandas as pd 
import polars as pl
