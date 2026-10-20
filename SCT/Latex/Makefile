# Makefile for SCT LaTeX Paper
# Usage:
#   make          — compile PDF (2 passes for cross-references)
#   make clean    — remove auxiliary files
#   make open     — open PDF in Preview

PAPER  = SCT_paper
LATEX  = pdflatex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all clean open

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex IEEEtran.cls
	$(LATEX) $(FLAGS) $(PAPER).tex
	$(LATEX) $(FLAGS) $(PAPER).tex
	@echo ""
	@echo "✓ PDF compiled: $(PAPER).pdf"

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out \
	      $(PAPER).toc $(PAPER).lof $(PAPER).lot \
	      $(PAPER).bbl $(PAPER).blg $(PAPER).synctex.gz

open: $(PAPER).pdf
	open $(PAPER).pdf
