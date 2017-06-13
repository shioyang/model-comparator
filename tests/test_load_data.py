import os,sys,inspect

# Add parent directory to sys.path in order to import ../*.py
currentdir = os.path.dirname( os.path.abspath(inspect.getfile(inspect.currentframe())) )
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from load_data import load_img_asarray

X = load_img_asarray('tests_data', 'jpg')
print(X)
