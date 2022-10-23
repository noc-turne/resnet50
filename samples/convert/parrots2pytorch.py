import pickle
import torch
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Convert Numpy Format'
                                 ' Model to Torch Model')
parser.add_argument('-i', '-I', '--input', type=str)
parser.add_argument('-o', '-O', '--output', type=str)
parser.add_argument('-to_np', '--to_numpy_model', action='store_true')
parser.add_argument('-to_tensor', '--to_tensor_model', action='store_true')


def to_numpy(v):
    if isinstance(v, torch.Tensor):
        result = v.cpu().detach().numpy()
    elif isinstance(v, (tuple, list)):
        result = []
        for i in range(len(v)):
            result.append(to_numpy(v[i]))
    elif isinstance(v, dict):
        result = {}
        for k, vv in v.items():
            result[k] = to_numpy(vv)
    elif isinstance(v, np.ndarray):
        print(
            'Caution! numpy data with size {} in input_file, '.format(v.shape)
            + 'this numpy object will translate to torch.tensor!',
            flush=True)
        result = v
    else:
        result = v
    return result


def to_tensor(v):
    if isinstance(v, np.ndarray):
        result = torch.from_numpy(v)
    elif isinstance(v, (tuple, list)):
        result = []
        for i in range(len(v)):
            result.append(to_tensor(v[i]))
    elif isinstance(v, dict):
        result = {}
        for k, vv in v.items():
            result[k] = to_tensor(vv)
    else:
        result = v
    return result


def numpy_2_tensor(input_file, output_file):
    np_dict = pickle.load(open(input_file, 'rb'))
    tensor_dict = to_tensor(np_dict)
    torch.save(tensor_dict, output_file)


def tensor_2_numpy(input_file, output_file):
    tensor_dict = torch.load(input_file, map_location='cpu')
    np_dict = to_numpy(tensor_dict)
    pickle.dump(np_dict, open(output_file, 'wb'))


def main():
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    assert input_file is not None, (
        "\n\tInput file name cannot be empty! use -i input_file")
    assert output_file is not None, (
        "\n\tOutput file name cannot be empty! use -o output_file")

    assert os.path.exists(input_file), (
        "\n\tInput File: {} is not exits!".format(input_file))
    assert not os.path.exists(output_file), (
        "\n\tOutput File: {} must not exits!".format(output_file))

    assert args.to_numpy_model or args.to_tensor_model, (
        '\n\tUse param -to_np to convert tensor model to '
        'numpy model'
        '\n\tUse param -to_tensor to covert numpy model '
        'to tensor model')

    if args.to_numpy_model:
        tensor_2_numpy(input_file, output_file)
    else:
        numpy_2_tensor(input_file, output_file)


if __name__ == '__main__':
    main()
