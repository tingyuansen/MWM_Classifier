from setuptools import setup

setup(name='MWM_Classifier',
      version='0.1',
      description='MWM_Broker',
      author='Yuan-Sen Ting',
      author_email='ting@ias.edu',
      license='MIT',
      url='https://github.com/tingyuansen/MWM_Classifier',
      package_dir = {},
      packages=['MWM_Classifier'],
      package_data={'MWM_Classifier':['neural_nets/*', 'spectra/*']},
      dependency_links = [],
      install_requires=['torch>=1.1', 'torchvision'])
