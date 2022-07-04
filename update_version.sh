echo "Updating version number in files: from $1 to $2"
sed -i "" -e "s/$1/$2/g" README.md
sed -i "" -e "s/$1/$2/g" setup.py
sed -i "" -e "s/$1/$2/g" ./doc/source/index.rst
sed -i "" -e "s/$1/$2/g" ./doc/source/conf.py
sed -i "" -e "s/$1/$2/g" ./PySeismoSoil/__init__.py
