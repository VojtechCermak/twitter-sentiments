from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='twitter-sentiments',
      version='0.1',
      description='Master thesis',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'Topic :: Sentiment classification',
      ],
      keywords='funniest joke comedy flying circus',
      url='https://github.com/VojtechCermak/twitter-sentiments',
      author='Vojtech Cermak',
      author_email='cermak.vojtech@seznam.cz',
      license='MIT',
      packages=['twitter-sentiments'],
      include_package_data=True,
      zip_safe=False)