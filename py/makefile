all:

run_1:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother richardson -omega 600" 1 1 "steady coarse richardson"

run_2:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 2 -smoother jacobicolor -nocoloring -fdeps 1.0" 1 2 "steady coarse jacobi (coloring off)"

run_3:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.2 -J 3 -Hmin 20.0 -smoother jacobicolor -fdeps 1.0" 1 3 "steady jacobicolor"

run_4:
	-@./testit.sh steady.py "-sweepsonly -monitor -irtol 0.5 -J 2 -smoother gsslow -fdeps 1.0" 1 4 "steady coarse gsslow"

run_5:
	-@./testit.sh steady.py "-sweepsonly -monitor -smoother newtonrs -J 4 -fdeps 2.0 -cyclemax 1 -band 2" 1 5 "steady newtonrs"

test_steady: run_1 run_2 run_3 run_4 run_5

test: test_steady

.PHONY: clean test_steady test run_1 run_2 run_3 run_4 run_5

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -f *.pyc *.pvd *.pvtu *.vtu *.m
	@rm -rf __pycache__ .pytest_cache/
