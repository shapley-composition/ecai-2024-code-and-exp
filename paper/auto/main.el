(TeX-add-style-hook
 "main"
 (lambda ()
   (setq TeX-command-extra-options
         "--synctex=1 -shell-escape")
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("cleveref" "capitalize" "noabbrev")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "bifurctree"
    "bifurctree1"
    "article"
    "art10"
    "microtype"
    "tikz"
    "pgfplots"
    "graphicx"
    "subcaption"
    "booktabs"
    "hyperref"
    "multirow"
    "icml2024"
    "amsmath"
    "amssymb"
    "mathtools"
    "amsthm"
    "bm"
    "cleveref")
   (TeX-add-symbols
    "theHalgorithm")
   (LaTeX-add-labels
    "sec:shapley"
    "eq:contrib"
    "eq:valuefunction"
    "sec:compo"
    "sec:ilr"
    "sec:shapcompo"
    "eq:valuefunctionsimplex"
    "sec:explain"
    "fig:3classesshap"
    "fig:3classesshapsum"
    "fig:3classes"
    "fig:4classesshapsum"
    "sec:balances"
    "fig:bifurc1"
    "fig:bifurc2"
    "fig:trees"
    "fig:moreclasses35"
    "fig:histiris"
    "fig:histmore"
    "sec:conclud"
    "app:properties"
    "eq:linearsimplex"
    "app:classcompo"
    "app:summarize"
    "tab:normiris"
    "tab:normdigit"
    "tab:projdigit"
    "app:algo"
    "alg:1"
    "alg:2"
    "app:correct")
   (LaTeX-add-bibliographies
    "biblio")
   (LaTeX-add-amsthm-newtheorems
    "theorem"
    "proposition"
    "lemma"
    "corollary"
    "definition"
    "assumption"
    "remark"))
 :latex)

