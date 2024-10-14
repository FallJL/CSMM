cd ..
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ../..