all:

run_1:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother richardson -omega 600" 1 1 "steady coarse richardson"

run_2:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother jacobicolor -cperthickness 1.0e10" 1 2 "steady coarse jacobi (slow)"

run_3:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 3 -Hmin 20.0 -smoother jacobicolor" 1 3 "steady jacobicolor"

run_4:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.5 -J 2 -smoother gsslow" 1 4 "steady coarse gsslow"

test_steady: run_1 run_2 run_3 run_4

test: test_steady

.PHONY: clean test_steady test run_1 run_2 run_3 run_4

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -f *.pyc *.pvd *.pvtu *.vtu *.m
	@rm -rf __pycache__ .pytest_cache/
