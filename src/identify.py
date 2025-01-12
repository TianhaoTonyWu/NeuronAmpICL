import torch
import json
import argparse
import os
import random

def process_input_file(input_path, percent):
    if input_path.split(".")[-1] == "pth":
        # Load LAPE matrix
        output = torch.load(input_path, weights_only=True)
        matrix = output['over_zero']
    else:
        # Load GV matrix
        with open(input_path, "r") as f:
            data = json.load(f)
        # Create a matrix from the JSON data
        matrix = torch.tensor(data)

    layers, number = matrix.shape
    flattened_matrix = matrix.view(-1)

    # Compute the number of top elements
    num_elements = flattened_matrix.numel()
    top_k = int(num_elements * percent)

    # Use torch.topk to find the largest `percent` of values
    # and their indices
    top_values, top_indices = torch.topk(flattened_matrix, top_k)

    # Convert flat indices back to 2D indices
    top_indices_2d = torch.stack([
        top_indices // number,
        top_indices % number
    ], dim=1)

    print(f"Top {round(percent * 100.0, 2)}% values:", top_values)
    print("Their indices in the original matrix:", top_indices_2d)

    output = [[[] for _ in range(layers)]]
    for i in top_indices_2d:
        l, c = i
        output[0][l].append(c.item())

    save_output = [[]]
    for j in output[0]:
        save_output[0].append(torch.tensor(j).type(torch.int64))

    return save_output

def generate_random_save_output(layers, max_indices_per_layer, total_elements, total_random_elements):
    """
    Generate a randomized save_output with the same structure as the original save_output.

    Args:
        layers (int): Number of layers.
        max_indices_per_layer (int): Maximum possible indices per layer.
        total_elements (int): The maximum number of indices available for random sampling in each layer.
        total_random_elements (int): Total number of elements to distribute randomly across all layers.

    Returns:
        list: A nested list of shape (1, layers, n), where n is a random number of indices for each layer.
    """
    random_output = [[]]
    elements_per_layer = [0] * layers

    # Distribute total_random_elements randomly among layers
    for _ in range(total_random_elements):
        layer = random.randint(0, layers - 1)
        if elements_per_layer[layer] < max_indices_per_layer:
            elements_per_layer[layer] += 1

    for layer_count in elements_per_layer:
        indices = random.sample(range(total_elements), layer_count)
        random_output[0].append(torch.tensor(indices).type(torch.int64))
    return random_output
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input_file", type=str,  default="/root/few_vs_zero/data/sni/task242/matrix/GV.json", help="Path to the input file.")
    parser.add_argument("-out", "--output_dir", type=str,  default="/root/few_vs_zero/data/sni/task242/activation_mask/Random/", help="Directory to save the output file.")
    parser.add_argument("-pt", "--percent", type=float, default=0.01, help="Percentage of top values to extract.")
    parser.add_argument("--random", action="store_true", help="Generate randomized save_output instead of processing the input file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = args.output_dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.random:
        # Generate randomized save_output
        layers = 32
        max_indices_per_layer = 11008
        total_elements = max_indices_per_layer
        total_rand_elements = int(layers * total_elements * 0.05)
        random_save_output = generate_random_save_output(layers, max_indices_per_layer, total_elements, total_random_elements=total_rand_elements)

        random_file_name =f"random{args.seed}.pth"
        torch.save(random_save_output, output_path + random_file_name)
        print("Randomized indices generated and saved.")
    else:
        # Process input file
        save_output = process_input_file(args.input_file, args.percent)
        file_name = f"{args.percent}p.pth"
        torch.save(save_output, output_path + file_name)
        print("Processed indices and saved.")

if __name__ == "__main__":
    main()
