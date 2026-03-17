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
        "color-brand-primary": "#129C7B",
        "color-brand-content": "#129C7B",
        "color-brand-visited": "#054C3B",
    },
    "dark_css_variables": {
        "color-brand-primary": "#04D78A",
        "color-brand-content": "#04D78A",
        "color-brand-visited": "#A7ED99",
        "color-foreground-primary": "#ffffff",
        "color-foreground-secondary": "#dedede",
        "color-foreground-muted": "#888888",
    },
    "light_logo": "esp-logotype-only-black.png",
    "dark_logo": "esp-logotype-only-white.png",  
}  
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/home-link.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_title = "Earth Species Project - AVEX Documentation"
html_static_path = ['_static']
html_css_files = ['custom.css'] 
html_favicon = "_static/favicon.svg"