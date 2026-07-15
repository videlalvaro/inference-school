PYTHON ?= python3
PANDOC ?= pandoc
LATEXMK ?= latexmk
BUILD_DIR := build
BOOK_NAME := Inference-School-Companion
MANUSCRIPT := $(BUILD_DIR)/manuscript.md
TEX := $(BUILD_DIR)/$(BOOK_NAME).tex
PDF := $(BUILD_DIR)/$(BOOK_NAME).pdf
CHAPTERS ?=
NOTE_FRAGMENTS := $(wildcard solution_notes/*.json)
LESSONS := $(wildcard ../Problems/*/README.md)
EXERCISE_SOURCES := $(shell find ../Sources/InferenceSchoolExercises -type f \( -name '*.swift' -o -name '*.metal' \))
SOLUTION_SOURCES := $(shell find ../Sources/InferenceSchoolSolutions -type f \( -name '*.swift' -o -name '*.metal' \))
APPENDICES := ../docs/ONE-TOKEN.md ../docs/MATH-PRIMER.md

.PHONY: all book check sample manuscript tex clean

all: book

book: $(PDF)
	@mkdir -p ../dist
	@cp $(PDF) ../dist/$(BOOK_NAME).pdf
	@printf 'Built %s\n' "../dist/$(BOOK_NAME).pdf"

check: book
	$(PYTHON) verify_book.py \
		--manuscript $(MANUSCRIPT) --tex $(TEX) \
		--log $(BUILD_DIR)/$(BOOK_NAME).log --pdf $(PDF) \
		--published ../dist/$(BOOK_NAME).pdf

sample:
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py --chapters 001 --output $(MANUSCRIPT)
	$(PANDOC) $(MANUSCRIPT) \
		--from markdown+raw_tex+raw_html+tex_math_dollars+fenced_divs+definition_lists+pipe_tables+task_lists \
		--standalone --top-level-division=chapter --syntax-highlighting=tango \
		--metadata-file=metadata.yaml --include-in-header=preamble.tex \
		--output=$(TEX)
	@cd $(BUILD_DIR) && $(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error $(BOOK_NAME).tex

$(MANUSCRIPT): generate_book.py frontmatter.md solution_notes.json $(NOTE_FRAGMENTS) $(LESSONS) $(EXERCISE_SOURCES) $(SOLUTION_SOURCES) $(APPENDICES)
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py $(if $(CHAPTERS),--chapters $(CHAPTERS),) --output $(MANUSCRIPT)

$(TEX): $(MANUSCRIPT) metadata.yaml preamble.tex
	$(PANDOC) $(MANUSCRIPT) \
		--from markdown+raw_tex+raw_html+tex_math_dollars+fenced_divs+definition_lists+pipe_tables+task_lists \
		--standalone --top-level-division=chapter --syntax-highlighting=tango \
		--metadata-file=metadata.yaml --include-in-header=preamble.tex \
		--output=$(TEX)

$(PDF): $(TEX)
	@cd $(BUILD_DIR) && $(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error $(BOOK_NAME).tex

clean:
	rm -rf $(BUILD_DIR)
