(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "--synctex=1 -shell-escape")
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-environments-local "TeXlstlisting")
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "figures/simplex"
    "figures/bifurctree"
    "figures/prob_simplex"
    "../paper/bifurctree1"
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
    "mathtools"
    "pgfplotstable"
    "biblatex"
    "soul"
    "pifont")
   (TeX-add-symbols
    "tikzxmark"
    "tikzcmark")
   (LaTeX-add-labels
    "eq:contrib"
    "fig:bifurctree")
   (LaTeX-add-bibliographies
    "../paper/biblio"))
 :latex)

