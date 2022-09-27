pip install flake8 wemake-python-styleguide==0.16.1 flake8-commas flake8-mutable flake8-length flake8-absolute-import
echo ""
echo "******************************"
echo ""
flake8 --config=flake8.cfg PySeismoSoil/
flake8 --config=flake8.cfg --ignore=D101,D102,N801 tests/
echo "----------------------"
flake8 --config=flake8.cfg --select WPS317 PySeismoSoil/
flake8 --config=flake8.cfg --select WPS317 tests/
