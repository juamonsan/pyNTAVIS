from setuptools import setup, find_packages

with open('../README.md') as readme_file:
    README = readme_file.read()
"""
with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()
"""
setup_args = dict(
    name='pyNAVIS',
    version='1.2.5+ntas.1',
    description='Useful tools to analyze and process spiking information from neuromorphic auditory sensors (NAS) and from the Neuromorphic Tri-Axial Sensor (NTAS). pyNTAVIS fork of pyNAVIS.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='GPL',
    packages=find_packages(),
    author='Juan P. Dominguez-Morales, Juan M. Montes-Sanchez (pyNTAVIS three-axis additions)',
    author_email='jpdominguez@us.es, juanmanuelmontes@us.es',
    keywords=['pyNAVIS', 'NAVIS', 'neuromorphic engineering', "neuromorphic audio processing", "neuromorphic sensors", 'pyNTAVIS', 'NTAS', 'tri-axial', 'accelerometer', 'magnetometer', 'spiking neural networks'],
    url='https://github.com/juamonsan/pyNTAVIS',
    project_urls={'Original pyNAVIS': 'https://github.com/jpdominguez/pyNAVIS'},
    # Original pyNAVIS on PyPI (https://pypi.org/project/pyNAVIS/) is the audio-only version, without three-axis (NTAS) support
    download_url='https://github.com/juamonsan/pyNTAVIS'
)

install_requires = [
	'matplotlib',
	'numpy',
	'scipy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)