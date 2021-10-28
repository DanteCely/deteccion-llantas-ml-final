import numpy


class Base:

    m_X = None
    m_y = None

    def __init__(self, in_x, in_y):
        assert isinstance(in_x, (list, numpy.matrix, numpy.ndarray)), \
            "Invalid X type."
        assert isinstance(in_y, (list, numpy.matrix, numpy.ndarray)), \
            "Invalid y type."

        if type(in_x) is list or type(in_x) is numpy.ndarray:
            self.m_X = numpy.matrix(in_x)
        else:
            self.m_X = in_x
        # end if
        if type(in_y) is list:
            self.m_y = numpy.matrix(in_y).T
        else:
            self.m_y = in_y
        # end if
        assert self.m_X.shape[0] == self.m_y.shape[0], "Invalid X,y sizes."
        assert self.m_y.shape[1] == 1, "Invalid y size."

        self.m_M = self.m_X.shape[0]
        self.m_N = self.m_X.shape[1]

    # end def

    def NumberOfExamples(self):
        return self.m_M

    # end def

    def VectorSize(self):
        return self.m_N + 1
    # end def

# end class

# eof - $RCSfile$
