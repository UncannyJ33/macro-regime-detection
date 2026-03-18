"""Transforms raw data into model-ready features.

Computes derived indicators (yield curve spread, rolling z-scores, MoM
changes), handles missing values, and runs PCA to reduce dimensionality
before regime modeling.
"""
