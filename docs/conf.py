#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Wilfried Mercier - LAM <wilfried.mercier@ilam.fr>

Configuration script for Sphinx documentation.
"""

from __future__ import annotations

import sys
import os

# Required for autodoc to work
sys.path.append(os.path.abspath('../..'))

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

###################################################
#               Project information               #
###################################################

project            = 'Clump finder'
copyright          = '2024 - 2025, Wilfried Mercier'
author             = 'Wilfried Mercier'
show_authors       = True

highlight_options  = {'default': {'lexers.python.PythonLexer'},
                     }

extensions         = ['sphinx.ext.autodoc',
                      'sphinx.ext.mathjax',
                      'sphinx.ext.viewcode',
                      'sphinx.ext.autosummary',
                      'sphinx.ext.intersphinx',
                      'jupyter_sphinx',
                      "sphinx_design",
                     ]

# The full version, including alpha/beta/rc tags
release            = '1.0'

#######################################################
#               Options for HTML output               #
#######################################################

html_theme = "pydata_sphinx_theme"
html_title = 'Clump finder'
html_context = {"default_mode" : "light"}

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            
            # URL where the link will redirect
            "url": "https://github.com/WilfriedMercier/clump_finder/tree/main",
            
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }],
    
    "logo": {
        
        # Alt text for blind people
        #"alt_text"    : "SFHandle documentation - Home",
        #"text"        : "SFHandle",
        #"image_light" : "_static/logo1.png",
        #"image_dark"  : "_static/logo1.png",
    },
    
    #"announcement"    : "",
    "show_nav_level"  : 2
}

html_collapsible_definitions = True
add_module_names = False

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

rst_prolog = """
.. role:: python(code)
  :language: python
  :class: highlight
  
.. _Colormap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
.. _ImageNormalize: https://docs.astropy.org/en/stable/api/astropy.visualization.ImageNormalize.html
.. _Cosmology: https://docs.astropy.org/en/stable/api/astropy.cosmology.Cosmology.html
.. _NDArray: https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray
"""