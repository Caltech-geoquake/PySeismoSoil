pip install flake8 wemake-python-styleguide flake8-commas flake8-mutable flake8-length flake8-absolute-import flake8-newspaper-style
echo ""
echo "******************************"
echo ""
flake8 --select NEWS100 PySeismoSoil/
# flake8 --config=setup.cfg PySeismoSoil/
echo "----------------------"
# flake8 --config=setup.cfg --ignore=D101,D102,N801 tests/
#echo "----------------------"
#flake8 --select WPS317 ./tests/*.py
