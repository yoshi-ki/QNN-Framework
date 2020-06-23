is_simple_core = False

if is_simple_core:
  from my_framework.core_simple import Variable
  from my_framework.core_simple import Function
  from my_framework.core_simple import using_config
  from my_framework.core_simple import no_grad
  from my_framework.core_simple import as_array
  from my_framework.core_simple import as_variable
  from my_framework.core_simple import setup_variable
else:
  from my_framework.core import Variable
  from my_framework.core import Function
  from my_framework.core import using_config
  from my_framework.core import no_grad
  from my_framework.core import as_array
  from my_framework.core import as_variable
  from my_framework.core import setup_variable
  from my_framework.core import Parameter
  from my_framework.models import Model
  from my_framework.layers import Layer
  from my_framework.optimizers import Optimizer

setup_variable()
