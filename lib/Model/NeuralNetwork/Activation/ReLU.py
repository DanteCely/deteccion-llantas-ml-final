import numpy
from .Base import *

'''
'''
class ReLU( Base ):

  '''
  '''
  def __init__( self ):
    pass
  # end def

  '''
  '''
  def __call__( self, z, derivative = False ):
    if derivative:
      return ( z >= 0.0 ).astype( z.dtype )
    else:
      return numpy.array( z ) * numpy.array( ( z >= 0.0 ).astype( z.dtype ) )
    # end if
  # end def

# end class

## eof - $RCSfile$
