import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import os

def load_model(checkpoint_path):
    # Load the configuration
    config = get_config("zoedepth", "infer")
    
    # Build the model
    model = build_model(config)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((384, 512)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_depth(model, input_image):
    with torch.no_grad():
        depth_prediction = model(input_image)
    return depth_prediction

def visualize_and_save_depth(depth_map, output_path):
    # Convert to numpy and remove batch dimension
    depth_map = depth_map.squeeze().cpu().numpy()
    
    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Create and save the visualization
    plt.figure(figsize=(10, 7.5))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('Predicted Depth Map')
    plt.savefig(output_path)
    plt.close()

def main():
    # Paths
    checkpoint_path = "/vol/fob-vol3/mi20/deghaisa/shortcuts/monodepth3_checkpoints/ZoeDepthv1_01-Aug_04-49-4c89a80f4070_latest.pt"
    image_path = "/vol/fob-vol3/mi20/deghaisa/suadd/inputs/1716b622a6cb42828e61a5546317a437-1616446329700003698.png"
    output_dir = "/vol/fob-vol3/mi20/deghaisa/code/zoe_out/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input image name
    input_filename = os.path.basename(image_path)
    output_filename = f"depth_{os.path.splitext(input_filename)[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Preprocess image
    input_image = preprocess_image(image_path)
    
    # Predict depth
    depth_prediction = predict_depth(model, input_image)
    
    # Visualize and save depth map
    visualize_and_save_depth(depth_prediction, output_path)
    
    print(f"Depth map saved to {output_path}")

if __name__ == "__main__":
    main()