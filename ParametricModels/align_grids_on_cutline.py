#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module provides a function to align grids on a cutline using ANSA software.
"""
import os
import ansa
from ansa import *
 
def align_grids_on_cutline():
    grids_set = base.GetEntity(constants.ABAQUS, "SET", 2)  # /SET ID = 2 holds grids.
    trgt_set = base.GetEntity(constants.ABAQUS, "SET", 1)  # SET ID = 1 holds target points
 
    base.AlignGrids(
        grids_set,
        trgt_set,
        align_to="grids",
        distance=100.0,
        offset=0.0,
        isospace_grids=True,
        copy_grids=False,
        redraw=False,
    )
    
#align_grids_on_cutline()    
