# README: Signed Graphs in Data Sciences via Communicability Geometry

This repository provides Python code used to analyze complex networks through communicability geometry, as introduced and developed in the paper:
F. Diaz-Diaz and E. Estrada, Signed graphs in data sciences via communicability geometry, Information Sciences, 122096, doi: https://doi.org/10.1016/j.ins.2025.122096.

Please cite the article above if you use this code in your research.

The main goal is to offer a set of computational tools for:
- Calculating various communicability metrics (signed communicability, distance, angles, etc.).
- Performing multidimensional scaling (MDS) on communicability-derived distances.
- Clustering nodes based on distance or angle metrics to obtain factions.
- Creating dendrograms revealing the hierarchy of alliances in the network.
- Visualizing networks with special attention to positive and negative edge weights.

Contents
1) communicability_geometry_applications.ipynb

A Jupyter Notebook demonstrating usage examples, from loading or constructing a graph all the way to computing and visualizing communicability measures.

2)utils.py

A Python module with utility functions to streamline analyses:
