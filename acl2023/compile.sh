#!/bin/bash
# LaTeX compilation script for acl2023.tex
# This paper now uses only custom.bib (not the 42MB anthology.bib)
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode acl2023.tex
bibtex acl2023
pdflatex -interaction=nonstopmode acl2023.tex
pdflatex -interaction=nonstopmode acl2023.tex
echo "Compilation complete!"
ls -lh acl2023.pdf
