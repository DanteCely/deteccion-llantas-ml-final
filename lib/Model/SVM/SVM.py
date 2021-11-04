import math

import numpy as np
from lib.Model.Regression.Base import Base


class SVM(Base):
    m_weights = None
    m_bias = None

    m_ValidTypes = (int, float, list, np.matrix, np.ndarray)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.m_weights = np.random.uniform(low=-1.0, high=1.0, size=(1, self.m_X.shape[1]))
        self.m_bias = float(0)

    def SetParameters(self, t):
        assert isinstance(t, self.m_ValidTypes), \
            'Invalid parameters type (' + str(type(t)) + ')'

        m_t = np.matrix(t).flatten()
        self.m_weights = m_t[:, 1:]
        self.m_bias = m_t[0, 0]

    def GetInputSize(self):
        if not self.m_weights is None:
            return self.m_weights.shape[1]
        else:
            return 0

    def GetParameters(self):
        t = np.zeros((1, self.GetInputSize() + 1))
        if not self.m_bias is None:
            t[0, 0] = self.m_bias
        # end if
        if not self.m_weights is None:
            t[:, 1:] = self.m_weights
        # end if
        return t

    def __call__(self, *args, **kwargs):
        assert len(args) > 0, 'No arguments passed to __call__()'

        if len(args) == 1:
            assert isinstance(args[0], self.m_ValidTypes)
            if isinstance(args[0], np.matrix):
                x = args[0]
            else:
                x = np.matrix(args[0])
            # end if
        else:
            x = np.matrix(list(args))

        return np.multiply(x, self.m_weights).sum(axis=1) + self.m_bias

    class Cost:

        m_Eps = 1e-8

        def __init__(self, m_model, batch_size=0):
            self.m_model = m_model

            if batch_size < 1:
                self.m_Batches = [(self.m_model.m_X, self.m_model.m_y)]
            else:
                m = self.m_model.m_X.shape[0]
                batch_count = int(math.ceil(float(m) / float(batch_size)))
                self.m_Batches = []
                for b in range(batch_count):
                    start = batch_size * b
                    end = start + batch_size
                    if end > m:
                        end = m
                    # end if
                    self.m_Batches += \
                        [(self.m_model.m_X[start: end, :], self.m_model.m_y[start: end, :])]

        def GetInitialParameters(self):
            return self.m_model.GetParameters()

        def GetModel(self):
            return self.m_model

        def GetNumberOfBatches(self):
            return len(self.m_Batches)

        def loss_matrix(self, batch):
            m_loss = np.multiply(self.m_model.m_weights, self.m_Batches[batch][0]).sum(axis=1) - self.m_model.m_bias
            m_loss = np.multiply(self.m_Batches[batch][1], m_loss)
            return m_loss

        def _Cost(self, theta, batch):
            j = math.inf
            self.m_model.SetParameters(theta)
            loss_matrix = self.loss_matrix(batch)

            cost = np.where(loss_matrix >= 1, 0, 1 - loss_matrix).mean()
            return cost, loss_matrix

        # Parameters:
        #  -- theta: new weights and bias
        #  -- batch: current batch index to process
        #  -- alpha: lambda factor for regularization
        def CostAndGradient(self, theta, batch):
            cost, loss_matrix = self._Cost(theta, batch)

            d_w = np.where(loss_matrix >= 1, 0, np.multiply(-self.m_Batches[batch][1], self.m_Batches[batch][0])).mean(axis=0)
            d_w = d_w.reshape((1, d_w.shape[0]))

            d_b = np.matrix(np.where(loss_matrix >= 1, 0, self.m_Batches[batch][1] * self.m_model.m_bias).mean())

            return [cost, np.concatenate((d_b, d_w), axis=1)]
