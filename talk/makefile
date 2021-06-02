all: talk.pdf

talk.pdf: talk.tex
	pdflatex talk
	pdflatex talk

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.dvi
