all: msc.pdf

msc.pdf: msc.tex msc.bib cropfigures
	pdflatex msc
	bibtex msc
	pdflatex msc
	pdflatex msc

_figures := stokesdomain idoaction
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/stokesdomain.pdf genfigs/idoaction.pdf &: genfigs/stokesdomain.py
	(cd genfigs/ && python3 stokesdomain.py)

.PHONY: clean writeout cropfigures

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -rf genfigs/*.pdf
	@rm -rf genfigs/__pycache__/ genfigs/poisson/__pycache__/ genfigs/sia/__pycache__/ genfigs/plap/__pycache__/
