PYTHON ?= python3
PANDOC ?= pandoc
LATEXMK ?= latexmk
BUILD_DIR := build
BOOK_NAME := Inference-School-Companion
MANUSCRIPT := $(BUILD_DIR)/manuscript.md
EPUB_MANUSCRIPT := $(BUILD_DIR)/manuscript.epub.md
EPUB_ASSET_DIR := $(BUILD_DIR)/epub-assets
EPUB_ASSET_STAMP := $(EPUB_ASSET_DIR)/.generated
TEX := $(BUILD_DIR)/$(BOOK_NAME).tex
PDF := $(BUILD_DIR)/$(BOOK_NAME).pdf
EPUB := $(BUILD_DIR)/$(BOOK_NAME).epub
SAMPLE_MANUSCRIPT := $(BUILD_DIR)/sample.md
SAMPLE_TEX := $(BUILD_DIR)/$(BOOK_NAME)-sample.tex
SAMPLE_PDF := $(BUILD_DIR)/$(BOOK_NAME)-sample.pdf
SAMPLE_EPUB_MANUSCRIPT := $(BUILD_DIR)/sample.epub.md
SAMPLE_EPUB_ASSET_DIR := $(BUILD_DIR)/sample-epub-assets
SAMPLE_EPUB := $(BUILD_DIR)/$(BOOK_NAME)-sample.epub
SOURCE_DATE_EPOCH ?= 1767225600
CHAPTERS ?=
NOTE_FRAGMENTS := $(wildcard solution_notes/*.json)
LESSONS := $(wildcard ../Problems/*/README.md)
EXERCISE_SOURCES := $(shell find ../Sources/InferenceSchoolExercises -type f \( -name '*.swift' -o -name '*.metal' \))
SOLUTION_SOURCES := $(shell find ../Sources/InferenceSchoolSolutions -type f \( -name '*.swift' -o -name '*.metal' \))
APPENDICES := ../docs/ONE-TOKEN.md ../docs/MATH-PRIMER.md
BOOK_INPUTS := generate_book.py frontmatter.md solution_notes.json $(NOTE_FRAGMENTS) $(LESSONS) $(EXERCISE_SOURCES) $(SOLUTION_SOURCES) $(APPENDICES)
EPUB_FROM := markdown+raw_html+tex_math_dollars+fenced_divs+definition_lists+pipe_tables+task_lists
EPUB_OPTIONS := \
	--from $(EPUB_FROM) --to epub3 --standalone \
	--top-level-division=chapter --split-level=1 --toc --toc-depth=2 \
	--syntax-highlighting=tango --mathml \
	--metadata-file=metadata.yaml \
	--metadata identifier=urn:inference-school:companion \
	--css=epub.css --resource-path=$(BUILD_DIR):.

.PHONY: all book epub test check sample sample-epub manuscript tex clean

all: book

book: $(PDF) $(EPUB)
	@mkdir -p ../dist
	@cp $(PDF) ../dist/$(BOOK_NAME).pdf
	@cp $(EPUB) ../dist/$(BOOK_NAME).epub
	@printf 'Built %s and %s\n' "../dist/$(BOOK_NAME).pdf" "../dist/$(BOOK_NAME).epub"

epub: $(EPUB)
	@printf 'Built %s\n' "$(EPUB)"

test:
	$(PYTHON) -m unittest discover -s . -p 'test_*.py'

check: test $(PDF) $(EPUB)
	$(PYTHON) verify_book.py \
		--manuscript $(MANUSCRIPT) --tex $(TEX) \
		--log $(BUILD_DIR)/$(BOOK_NAME).log --pdf $(PDF) \
		--published ../dist/$(BOOK_NAME).pdf \
		--epub $(EPUB) --published-epub ../dist/$(BOOK_NAME).epub

sample:
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py --chapters 001 --output $(SAMPLE_MANUSCRIPT)
	$(PANDOC) $(SAMPLE_MANUSCRIPT) \
		--from markdown+raw_tex+raw_html+tex_math_dollars+fenced_divs+definition_lists+pipe_tables+task_lists \
		--standalone --top-level-division=chapter --syntax-highlighting=tango \
		--metadata-file=metadata.yaml --include-in-header=preamble.tex \
		--output=$(SAMPLE_TEX)
	@cd $(BUILD_DIR) && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error $(BOOK_NAME)-sample.tex
	@printf 'Built %s\n' "$(SAMPLE_PDF)"

sample-epub:
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py --format epub --chapters 001 \
		--asset-directory $(SAMPLE_EPUB_ASSET_DIR) \
		--output $(SAMPLE_EPUB_MANUSCRIPT)
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PANDOC) $(SAMPLE_EPUB_MANUSCRIPT) \
		$(EPUB_OPTIONS) --output=$(SAMPLE_EPUB)
	@printf 'Built %s\n' "$(SAMPLE_EPUB)"

$(MANUSCRIPT): $(BOOK_INPUTS)
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py $(if $(CHAPTERS),--chapters $(CHAPTERS),) --output $(MANUSCRIPT)

$(EPUB_MANUSCRIPT): $(BOOK_INPUTS)
	@mkdir -p $(BUILD_DIR)
	$(PYTHON) generate_book.py --format epub \
		$(if $(CHAPTERS),--chapters $(CHAPTERS),) \
		--asset-directory $(EPUB_ASSET_DIR) --output $(EPUB_MANUSCRIPT)

$(EPUB_ASSET_STAMP): $(EPUB_MANUSCRIPT)
	@test -f $(EPUB_ASSET_STAMP) || $(PYTHON) generate_book.py --format epub \
		$(if $(CHAPTERS),--chapters $(CHAPTERS),) \
		--asset-directory $(EPUB_ASSET_DIR) --output $(EPUB_MANUSCRIPT)
	@touch $(EPUB_ASSET_STAMP)

$(TEX): $(MANUSCRIPT) metadata.yaml preamble.tex
	$(PANDOC) $(MANUSCRIPT) \
		--from markdown+raw_tex+raw_html+tex_math_dollars+fenced_divs+definition_lists+pipe_tables+task_lists \
		--standalone --top-level-division=chapter --syntax-highlighting=tango \
		--metadata-file=metadata.yaml --include-in-header=preamble.tex \
		--output=$(TEX)

$(PDF): $(TEX)
	@cd $(BUILD_DIR) && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error $(BOOK_NAME).tex

$(EPUB): $(EPUB_MANUSCRIPT) $(EPUB_ASSET_STAMP) metadata.yaml epub.css
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(PANDOC) $(EPUB_MANUSCRIPT) \
		$(EPUB_OPTIONS) --output=$(EPUB)

clean:
	rm -rf $(BUILD_DIR)
