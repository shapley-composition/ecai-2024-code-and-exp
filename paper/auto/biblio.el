(TeX-add-style-hook
 "biblio"
 (lambda ()
   (setq TeX-command-extra-options
         "--synctex=1 -shell-escape")
   (LaTeX-add-bibitems
    "NIPS2017_7062"
    "vstrumbelj2014explaining"
    "datta2016"
    "shapley1953value"
    "pawlowskymodeling"
    "aitchison1982"
    "aitchison2001"
    "egozcue2003isometric"))
 '(or :bibtex :latex))

