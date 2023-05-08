#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@License: (C) Copyright 2013-2022.
@Desc:
"""

import numpy as np
import pandas as pd
import cv2
import shapely.geometry as shp
from shapely.geometry import LineString, GeometryCollection, Point, Polygon
from shapely.ops import split, snap


def find_boundary(im_map_arr, 
                 min_thre_binary=None, 
                 max_thre_binary=None, 
                 gaussian_sigma=0, 
                 canny_thre1=200, 
                 canny_thre2=255,
                 se_ksize=9):
    """
    find all contuors using open cv
    All contours are returned by sorting their areas, descreasing. 
    
    return
    -----
    contuors areas and contuors objects 
    
    """
    
    a = im_map_arr.max()
    if min_thre_binary is None:
        min_thre_binary = a/2
    if max_thre_binary is None:
        max_thre_binary = a
    
    _, thresh = cv2.threshold(im_map_arr, 
                              min_thre_binary, 
                              max_thre_binary, 
                              cv2.THRESH_BINARY)#cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    blurred = cv2.GaussianBlur(thresh, (3, 3), gaussian_sigma)
    tight_edged = cv2.Canny(blurred, canny_thre1,canny_thre2)
    
    # find the contours in the dilated image
    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_ksize, se_ksize))
    # apply the dilation operation to the edged image
    dilate = cv2.dilate(tight_edged, kernel, iterations=1) #
    # dilate = cv2.morphologyEx(tight_edged, cv2.MORPH_DILATE, kernel)
    # find the contours in the dilated image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError('No boundary found')
    
    r_areas = [cv2.contourArea(c) for c in contours]
    sorted_area = np.sort(r_areas)[::-1]
    sorted_area_ind = np.argsort(r_areas)[::-1]
    
    sorted_contour = []
    for i in sorted_area_ind:
        sorted_contour.append(contours[i])
    
    return sorted_area, sorted_contour


def smooth_boundary(im_map_arr, boundary, factor=0.01,sm_ksize=25):
    """
    smooth all the boundarys
    
    parameters
    -----
    boudarys: list
        list of countours returned by cv2.findContours
    
    factor: float
        smoothing factor, the bigger, the smoother
    
    """
    im_map = np.zeros(im_map_arr.shape[:2], dtype="uint8") * 255
    smoothened=[]
    for c in boundary:
        epsilon = factor*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        # further smooth
        cv2.drawContours(im_map, [approx], -1, 255, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sm_ksize,sm_ksize))
        dilate = cv2.morphologyEx(im_map, cv2.MORPH_CLOSE, kernel)
        
        # make edge outline
        edge = cv2.Canny(dilate, 10, 255)
        # thicken edge
        edge = cv2.GaussianBlur(edge, (3,3), 0.1)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        r_areas = [cv2.contourArea(c) for c in contours]
        sorted_area = np.sort(r_areas)[::-1]
        sorted_area_ind = np.argsort(r_areas)[::-1]
        
        smoothened.append(contours[sorted_area_ind[0]]) # max one
        
    return smoothened


""""
Split a complex linestring using shapely.

Inspired by https://github.com/Toblerity/Shapely/issues/1068

reference: https://gis.stackexchange.com/questions/387149/shapely-strange-splits-when-splitting-linestring-and-polygon
"""

def complex_split(geom: LineString, splitter):
    """Split a complex linestring by another geometry without splitting at
    self-intersection points.

    Parameters
    ----------
    geom : LineString
        An optionally complex LineString.
    splitter : Geometry
        A geometry to split by.

    Warnings
    --------
    A known vulnerability is where the splitter intersects the complex
    linestring at one of the self-intersecting points of the linestring.
    In this case, only the first path through the self-intersection will
    be split.

    Examples
    --------
    >>> complex_line_string = LineString([(0, 0), (1, 1), (1, 0), (0, 1)])
    >>> splitter = LineString([(0, 0.5), (0.5, 1)])
    >>> complex_split(complex_line_string, splitter).wkt
    'GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1, 1 0, 0.25 0.75), LINESTRING (0.25 0.75, 0 1))'

    Return
    ------
    GeometryCollection
        A collection of the geometries resulting from the split.
    """
    if geom.is_simple:
        return split(geom, splitter)
    
    if isinstance(splitter, Polygon):
        splitter = splitter.exterior

    # Ensure that intersection exists and is zero dimensional.
    relate_str = geom.relate(splitter)
    if relate_str[0] == '1':
        raise ValueError('Cannot split LineString by a geometry which intersects a '
                         'continuous portion of the LineString.')
    if not (relate_str[0] == '0' or relate_str[1] == '0'):
        return geom

    intersection_points = geom.intersection(splitter)
    # This only inserts the point at the first pass of a self-intersection if
    # the point falls on a self-intersection.
    snapped_geom = snap(geom, intersection_points, tolerance=1.0e-12)  # may want to make tolerance a parameter.
    # A solution to the warning in the docstring is to roll your own split method here.
    # The current one in shapely returns early when a point is found to be part of a segment.
    # But if the point was at a self-intersection it could be part of multiple segments.
    return split(snapped_geom, intersection_points)


if __name__ == '__main__':
    complex_line_string = LineString([(0, 0), (1, 1), (1, 0), (0, 1)])
    splitter = LineString([(0, 0.5), (0.5, 1)])

    out = complex_split(complex_line_string, splitter)
    print(out)
#     assert len(out) == 2

    # test inserting and splitting at self-intersection
    pt = Point(0.5, 0.5)
    print(f'snap: {snap(complex_line_string, pt, tolerance=1.0e-12)}')
    print(f'split: {split(snap(complex_line_string, pt, tolerance=1.0e-12), pt)}')