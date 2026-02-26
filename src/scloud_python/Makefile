PYTEST = pytest


C_TESTS = c_encode_tests.py c_sample_tests.py c_pke_tests.py
J_TESTS = j_encode_tests.py j_sample_tests.py j_keccak_tests.py j_matrix_tests.py j_pke_tests.py

all: jasmin_unit_tests

c_unit_tests: $(C_TESTS)
	 $(PYTEST) $(C_TESTS)

jasmin_unit_tests: $(J_TESTS)
	 $(PYTEST) $(J_TESTS)
