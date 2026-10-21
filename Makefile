figures:
	pytest --viz
	mkdir -p docs/figures
	cp tests/artifacts/*.png docs/figures/

.PHONY: figures
