from setuptools import setup

with open('README.md', 'r') as f:
  long_description = f.read()

def get_git_version():
  def call_git_describe():
    try:
      from subprocess import Popen, PIPE
      p = Popen(['git', 'describe', '--abbrev=0', '--tags'], stdout=PIPE, stderr=PIPE)
      p.stderr.close()
      line = p.stdout.readlines()[0]
      line = line.strip()
      if line.startswith('v'):
        line = line[1:]
      return line
    except:
      return None
  version = call_git_describe()
  if version is None:
    version = "0.0.0"
  return version

setup(
  name='stylegan2-latent-tool',
  version=get_git_version(),
  author='Jeff Sontag',
  url='https://github.com/jmoso13/StyleGAN2-Latent-Tool',
  description='Tool for Inspecting StyleGAN2 Latent Space',
  long_description=long_description,
  packages=[
    'latent_utils'
  ],
  install_requires=[
    'tensorflow==1.15',
    'numpy'
    ],
  extras_require={

  },
  dependency_links=[

  ],
  include_package_data=True,
)