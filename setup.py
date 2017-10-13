from setuptools import setup

setup(name='feature_importance_gbm',
      version='0.1',
      description='Computation of feature importance for gradient boosting ensemble methods',
      url='https://github.com/Cnstant/feature_importance_gbm',
      author='Constant Bridon',
      author_email='constant.bridon@example.com',
      license='MIT',
      packages=['feature_importance_gbm'],
      zip_safe=False,
      package_dir = {'feature_importance_gbm':'feature_importance_gbm'},
      include_package_data = True,
      setup_requires = ['pytest-runner'],
      tests_require = ['pytest']
      )