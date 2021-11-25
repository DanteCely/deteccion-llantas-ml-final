import numpy
import sys

MAX_PERCENTAGE = 0.9999999
fine_name = 'all_data'


def write_data(data, filename):
    file_object = open(f'./{filename}.csv', 'a')
    file_object.write(data)
    file_object.close()


def get_keep_columns(vSPercentages, keep):
    sum_S = 0
    keep_columns = 1

    for v in vSPercentages:
        sum_S += v
        if sum_S < keep:
            keep_columns += 1

    return sum_S, keep_columns


def get_the_best_columns(input_file, columns_from, columns_kernel):
    # read and shuffle data
    # 57600
    columns_to = columns_from + columns_kernel
    Data = input_file[:, columns_from: columns_to]
    numpy.random.shuffle(Data)
    print('Shape:', Data.shape)

    # Gaussian Characterization
    m = Data.mean(axis=0)
    S \
      = (Data - m).T @ (Data - m) / (Data.shape[0] - 1)

    print('Eigen Values and eigen Vectors')

    eigen_values, eigen_vectors = numpy.linalg.eig(S)

    iS = eigen_values.argsort()[:: -1]
    eigen_values = eigen_values[iS]
    eigen_vectors = eigen_vectors[:, iS]
    sumS = eigen_values.sum()

    vSPercentages = eigen_values / sumS

    sum_S, keep_columns = get_keep_columns(vSPercentages, MAX_PERCENTAGE)

    print('columns taken:', f'{columns_from} - {columns_to}')
    print('sum_S:\n', sum_S)
    print('keep_columns:\n', keep_columns)
    to_print = f'{MAX_PERCENTAGE},{columns_from},{columns_to},{sum_S},{keep_columns}\n'

    write_data(to_print, f'pca_analysis_{fine_name}')


if __name__ == "__main__":
    print('============| Getting Data...')
    input_file = numpy.genfromtxt(sys.argv[1], delimiter=',')
    input_file = input_file[:, 0: -1]  # remove last column
    print('============| Done.')
    input_file_tam = input_file.shape[1]
    print('============| Start Analysis |============')

    # for i in range(0,input_file_tam, 1000):

    get_the_best_columns(input_file, 100000, 5000)

    # print('Mean', m)
    # print('Covariance', S)

    # print('Inertia Matrix')
    # D2 = ( D - m ) ** 2
    # D2s = D2.sum()
    # D2TD2 = D2.T @ D2
    # np.fill_diagonal( D2TD2, 0 )
    # I = np.diag( D2 - D2.sum( axis = 0 ) ) - D2TD2

    # Inertia values and inercia vectors
    # Ivalues, Ivectors = numpy.linalg.eig( S )
    # print('Inercia:\n', I)
