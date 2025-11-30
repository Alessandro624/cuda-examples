# Top-level Makefile for cuda examples
# Targets: all (default), build, clean, list

SUBDIRS := $(shell for d in */; do [ -f "$${d}Makefile" ] && printf "%s " "$${d}"; done)

.PHONY: all build clean list help
all: build

build:
	@echo "Building subprojects: $(SUBDIRS)"
	@for d in $(SUBDIRS); do \
		echo "== Building $$d =="; \
		$(MAKE) -C $$d || exit 1; \
	done

clean:
	@for d in $(SUBDIRS); do \
		echo "== Cleaning $$d =="; \
		$(MAKE) -C $$d clean || true; \
	done

list:
	@echo "Detected subprojects:" $(SUBDIRS)

help:
	@echo "Usage: make [target]"
	@echo "Targets: all (default), build, clean, list, help"
