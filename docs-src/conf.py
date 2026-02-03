# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AVEX'
copyright = '2026, Earth Species Project'
author = 'Earth Species Project'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',    
    'sphinx_copybutton'
]

myst_enable_extensions = [
    "colon_fence",   
    "deflist",
    "substitution",
]
myst_heading_anchors = 3
myst_all_links_external = True
myst_links_external_new_tab = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo' 
 
html_theme_options = { 
    "sidebar_hide_name": True,  
    "source_view_link": "https://github.com/earthspecies/avex",
    "light_css_variables": {
        "color-brand-primary": "#007388",
        "color-brand-content": "#007388",
        "color-brand-visited": "#7BC4AD",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ecad6",
        "color-brand-content": "#5ecad6",
    },
    "light_logo": "logo_light_mode.png",
    "dark_logo": "logo_dark_mode.png",  
}  
html_title = "Earth Species Project - AVEX Documentation"
html_static_path = ['_static']
html_css_files = ['custom.css'] 
html_favicon = "_static/favicon.svg"