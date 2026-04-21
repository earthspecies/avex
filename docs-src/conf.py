# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import json
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AVEX'
copyright = '2026, Earth Species Project'
author = 'Earth Species Project'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_design',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "attrs_inline",
]
myst_heading_anchors = 3
myst_all_links_external = True
myst_links_external_new_tab = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

# Notebooks: use committed outputs, do not re-execute
nb_execution_mode = "off"
nb_mime_priority_overrides = [
    ("html", "application/vnd.plotly.v1+json", 0),
    ("html", "text/html", 1),
]

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
html_js_files = ['custom.js']
html_favicon = "_static/favicon.svg"


def add_tutorial_meta(app, pagename, templatename, context, doctree):
    """Inject notebook tutorial metadata into the Jinja2 template context."""
    src_path = os.path.join(app.srcdir, pagename + '.ipynb')
    if os.path.exists(src_path):
        try:
            with open(src_path) as f:
                nb = json.load(f)
            context['tutorial_meta'] = nb.get('metadata', {}).get('tutorial')
            return
        except Exception:
            pass
    context['tutorial_meta'] = None


def override_notebook_toc_title(app, doctree):
    """Override the TOC/sidebar title for notebooks with a sidebar_title in metadata."""
    from docutils import nodes
    docname = app.env.docname
    src_path = os.path.join(app.srcdir, docname + '.ipynb')
    if os.path.exists(src_path):
        try:
            with open(src_path) as f:
                nb = json.load(f)
            sidebar_title = nb.get('metadata', {}).get('tutorial', {}).get('sidebar_title')
            if sidebar_title:
                app.env.titles[docname] = nodes.title('', sidebar_title)
        except Exception:
            pass


def setup(app):
    app.connect('html-page-context', add_tutorial_meta)
    app.connect('doctree-read', override_notebook_toc_title)