LATEX=pdflatex
BIBTEX=bibtex
REPORT_DIR=report
MAIN=main

.PHONY: all report clean

all: report

# Compile the LaTeX report
# Runs pdflatex -> bibtex -> pdflatex x2 to ensure references are resolved
report:
	cd $(REPORT_DIR) && $(LATEX) $(MAIN).tex
	cd $(REPORT_DIR) && $(BIBTEX) $(MAIN)
	cd $(REPORT_DIR) && $(LATEX) $(MAIN).tex
	cd $(REPORT_DIR) && $(LATEX) $(MAIN).tex
	cd $(REPORT_DIR) && rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz $(MAIN).toc final_report.pdf

# Clean intermediate LaTeX build files
clean:
	rm -f $(REPORT_DIR)/*.aux \
	      $(REPORT_DIR)/*.log \
	      $(REPORT_DIR)/*.out \
	      $(REPORT_DIR)/*.bbl \
	      $(REPORT_DIR)/*.blg \
	      $(REPORT_DIR)/*.fls \
	      $(REPORT_DIR)/*.fdb_latexmk \
	      $(REPORT_DIR)/*.synctex.gz \
	      $(REPORT_DIR)/*.toc \
	      $(REPORT_DIR)/*.pdf
