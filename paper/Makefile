all : index.tex
	pdflatex $<
	bibtex `echo $< | cut -d'.' -f 1` || echo ""
	pdflatex $<
	pdflatex $<

clean :
	rm -rf index.pdf *.log *.aux *.bbl *.blg *.toc *.tex~ *.out
