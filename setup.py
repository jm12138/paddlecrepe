from setuptools import setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='paddlecrepe',
    description='PaddlePaddle implementation of CREPE pitch tracker',
    version='0.0.17',
    author='Jm12138',
    author_email='2286040843@qq.com',
    url='https://github.com/jm12138/paddlecrepe',
    install_requires=['librosa==0.9.1', 'resampy', 'scipy', 'tqdm'],
    packages=['paddlecrepe'],
    package_data={'paddlecrepe': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pitch', 'audio', 'speech', 'music', 'paddlepaddle', 'crepe'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
