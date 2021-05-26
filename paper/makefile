all: msg.pdf

msg.pdf: msg.tex msg.bib cropfigures
	pdflatex msg
	bibtex msg
	pdflatex msg
	pdflatex msg

_figures := stokesdomain idoaction extruded
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/stokesdomain.pdf genfigs/idoaction.pdf &: genfigs/stokesdomain.py
	(cd genfigs/ && python3 stokesdomain.py)

genfigs/extruded.pdf &: genfigs/extruded.py
	(cd genfigs/ && python3 extruded.py)

.PHONY: clean writeout cropfigures

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -rf genfigs/*.pdf
	@rm -rf genfigs/__pycache__/ genfigs/poisson/__pycache__/ genfigs/sia/__pycache__/ genfigs/plap/__pycache__/
