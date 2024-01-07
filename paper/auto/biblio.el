(TeX-add-style-hook
 "biblio"
 (lambda ()
   (setq TeX-command-extra-options
         "--synctex=1 -shell-escape")
   (LaTeX-add-bibitems
    "NIPS2017_7062"
    "aitchison1980"
    "vstrumbelj2014explaining"
    "datta2016"
    "shapley1953value"
    "pawlowskymodeling"
    "aitchison1982"
    "aitchison2001"
    "egozcue2003isometric"
    "egozcue2005groups"
    "pedregosa2011scikit"
    "aitchison1990relative"
    "egozcue2011evidence"
    "egozcue2018evidence"
    "noe2023representing"
    "angelov2021explainable"))
 '(or :bibtex :latex))

