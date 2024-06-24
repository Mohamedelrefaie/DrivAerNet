#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module provides functions to calculate the projected frontal area using ANSA software.
"""

def projected_frontal_area_m2():
    """
    Calculates the projected frontal area in square meters using ANSA software.

    This function calculates the projected frontal area of a model, converts it from square millimeters to square meters,
    and writes the result to a text file.

    Args:
        None

    Returns:
        None
    """
    ret = mesh.Outline(1, 0, 0, outline_internal_detected_voids=True)

    # Convert frontal area from mm² to m²
    frontal_area_m2 = ret.frontal_area_filled / 1e6  # 1 mm² = 1e-6 m²

    # Open the text file for appending
    with open("frontal_area.txt", "w") as file:
        # Write the converted frontal area in square meters
        file.write(f"Frontal area filled: {frontal_area_m2:.6f} m²\n")

#if __name__ == "__main__":
    #projected_frontal_area_m2()
