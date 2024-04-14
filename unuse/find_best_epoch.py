def find_best_epoch(filename):
    best_epoch = None
    max_value = float('-inf')
    
    with open(filename, 'r') as file:
        for line in file:
            if 'INFO:EPOCH:' in line:
                current_epoch = int(line.split()[-1])
            if 'NDCG@20, IoU=0.7' in line:
                # Skip to the next line where the values are
                values_line = next(file).strip()
                # Assuming 'NDCG@20, IoU=0.7' is the 6th value in the list
                ndcg_20_iou_7_value = float(values_line.split()[5])
                
                test_values_line = next(file).strip()
                performance = [element for pair in zip(values_line.split()[1:], test_values_line.split()[1:])  for element in pair]
                performance = [float(i) for i in performance]
                if ndcg_20_iou_7_value > max_value:
                    max_value = ndcg_20_iou_7_value
                    print(max_value)
                    best_epoch = current_epoch
                    print(performance)
    
    return best_epoch

# Example usage
filename = "results/ReLoCLNet_top_40_20240326_163257/20240326_163257_ReLoCLNet_top_40.log"
best_epoch = find_best_epoch(filename)
print("The best epoch for NDCG@20, IoU=0.7 is:", best_epoch)