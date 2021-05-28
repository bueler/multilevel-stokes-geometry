all:

run_1:
	-@./testit.sh steady.py "-sweepsonly -irtol 2.0e-1 -J 2 -smoother richardson -omega 600" 1 1 "steady coarse richardson"

run_2:
	-@./testit.sh steady.py "-sweepsonly -irtol 2.0e-1 -J 2 -smoother jacobislow" 1 2 "steady coarse jacobislow"

test_steady: run_1 run_2

test: test_steady

.PHONY: clean test_steady test run_1 run_2

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -f *.pyc *.pvd *.pvtu *.vtu *.m
	@rm -rf __pycache__ .pytest_cache/
