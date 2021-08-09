pip install --target packaging -r requirements.txt --upgrade
rm packaging.zip
cd packaging
zip -r ../packaging.zip .
cd ..
zip -g packaging.zip function.py