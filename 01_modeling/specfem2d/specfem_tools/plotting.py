"""Compatibility layer. Generic gather plotting moved to segy_tools.plotting.
SPECFEM model geometry plotting lives in specfem_tools.model.
"""
from segy_tools.plotting import *
from .model import plot_interface_geometry
