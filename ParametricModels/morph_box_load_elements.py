#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module provides functions to load orthogonal morphing boxes using ANSA software.
"""

def Load_Ortho_Boxes():
    """
    Loads orthogonal morphing boxes using ANSA software.

    This function collects shell and morphing box entities and then applies morphing loads to these entities.
    It uses different configurations to load the morphing boxes based on their visibility and database settings.

    Args:
        None

    Returns:
        None
    """
    shells = base.CollectEntities(constants.NASTRAN, None, "SHELL", filter_visible=True)
    morphes = base.CollectEntities(constants.NASTRAN, None, "MORPHBOX", filter_visible=True)
    morph.MorphLoad(morphes, entities_to_load=shells)
    morph.MorphLoad(morphes, db_or_visib="Visib")
    morph.MorphLoad(morphes, None, "Visib")
