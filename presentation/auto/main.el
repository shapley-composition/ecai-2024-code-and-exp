(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "--synctex=1 -shell-escape")
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "pgfplots"
    "tikz"
    "amssymb"
    "amsmath"
    "mathrsfs"
    "stmaryrd"
    "bm"
    "amsthm"
    "amsfonts"
    "pgfplotstable"
    "biblatex"
    "pifont")
   (TeX-add-symbols
    "tikzxmark"
    "tikzcmark")
   (LaTeX-add-labels
    "eq:contrib")
   (LaTeX-add-bibliographies
    "../paper/biblio"))
 :latex)

