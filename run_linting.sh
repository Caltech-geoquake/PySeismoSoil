pip install flake8 wemake-python-styleguide flake8-commas flake8-mutable flake8-length flake8-absolute-import
echo ""
echo "******************************"
echo ""
flake8 ./PySeismoSoil/helper_hh_model.py
echo "----------------------"
flake8 --select WPS317 ./PySeismoSoil/helper_hh_model.py
