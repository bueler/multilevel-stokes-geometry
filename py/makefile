all:

run_1:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother richardson -omega 600" 1 1 "steady coarse richardson"

run_2:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother jacobislow" 1 2 "steady coarse jacobislow"

run_3:
	-@./testit.sh steady.py "-sweepsonly -monitor -cyclemax 2 -J 3 -smoother gsslow -mz 2" 1 3 "steady gsslow"

test_steady: run_1 run_2 run_3

test: test_steady

.PHONY: clean test_steady test run_1 run_2

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -f *.pyc *.pvd *.pvtu *.vtu *.m
	@rm -rf __pycache__ .pytest_cache/
