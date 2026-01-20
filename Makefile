.PHONY: install

install:
	uv venv --python 3.9
	uv pip install pip
	uv pip install wheel setuptools build
	
	cd lib/SPFlow/src && bash create_pip_dist.sh 
	uv pip install lib/SPFlow/src/dist/spflow-0.0.40-py3-none-any.whl

	# 3. NASLib
	uv pip install numpy Cython
	cd NASLib && uv pip install --upgrade uv pip setuptools wheel
	cd NASLib && uv pip install --no-build-isolation -e .
	cd NASLib && python setup.py bdist_wheel
	uv pip install NASLib/dist/naslib-0.1.0-py3-none-any.whl

	# 4. Benchmarks
	cd HPOBench && uv pip install .
	cd jahs_bench_201 && uv pip install -e .

	# 5. Requirements
	uv pip install -r requirements.txt
	
	# 6. ConfigSpace
	uv pip install ConfigSpace

	# 7. Final Setup
	chmod +x setup.sh
	./setup.sh