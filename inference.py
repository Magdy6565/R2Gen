import os  
import torch  
import pickle  
import argparse  
import glob  
from PIL import Image  
import torchvision.transforms as transforms  
from models.r2gen import R2GenModel  
  
def parse_args():    
    parser = argparse.ArgumentParser(description='Run inference with R2Gen model on multiple images')    
        
    # Model and tokenizer paths    
    parser.add_argument('--model_path', type=str, required=True,     
                        help='Path to the saved model checkpoint')    
    parser.add_argument('--tokenizer_path', type=str, required=True,     
                        help='Path to the saved tokenizer')    
        
    # Image paths - now with options for batch processing  
    parser.add_argument('--image_path', type=str, default=None,     
                        help='Path to a single input image (for backward compatibility)')  
    parser.add_argument('--image_dir', type=str, default=None,  
                        help='Directory containing multiple images to process')  
    parser.add_argument('--image_list', type=str, default=None,  
                        help='Text file with a list of image paths, one per line')  
    parser.add_argument('--output_dir', type=str, default='./reports',  
                        help='Directory to save generated reports')  
    parser.add_argument('--batch_size', type=int, default=4,  
                        help='Batch size for processing multiple images')  
        
    # Dataset type    
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],    
                        help='Dataset type (affects image processing)')    
        
    # Generation parameters    
    parser.add_argument('--sample_method', type=str, default='beam_search',     
                        help='the sample methods to sample a report.')    
    parser.add_argument('--beam_size', type=int, default=3,     
                        help='Beam size for beam search')    
    parser.add_argument('--max_seq_length', type=int, default=60,     
                        help='Maximum sequence length for generation')    
    parser.add_argument('--block_trigrams', type=int, default=1,    
                        help='Whether to block trigram repetitions')    
    parser.add_argument('--temperature', type=float, default=1.0,     
                        help='the temperature when sampling')    
    parser.add_argument('--sample_n', type=int, default=1,     
                        help='the sample number per image')    
    parser.add_argument('--group_size', type=int, default=1,     
                        help='the group size')    
    parser.add_argument('--output_logsoftmax', type=int, default=1,     
                        help='whether to output the probabilities')    
    parser.add_argument('--decoding_constraint', type=int, default=0,     
                        help='whether decoding constraint')    
        
    # Model parameters (should match the saved model)    
    parser.add_argument('--d_model', type=int, default=512,     
                        help='Dimension of the model')    
    parser.add_argument('--d_ff', type=int, default=512,     
                        help='Dimension of the feed-forward layer')    
    parser.add_argument('--d_vf', type=int, default=2048,     
                        help='Dimension of the visual features')    
    parser.add_argument('--num_heads', type=int, default=8,     
                        help='Number of attention heads')    
    parser.add_argument('--num_layers', type=int, default=3,     
                        help='Number of transformer layers')    
    parser.add_argument('--dropout', type=float, default=0.1,     
                        help='Dropout rate')    
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,     
                        help='the dropout rate of the output layer')    
    parser.add_argument('--logit_layers', type=int, default=1,     
                        help='the number of the logit layer')    
        
    # Relational Memory parameters    
    parser.add_argument('--rm_num_slots', type=int, default=3,     
                        help='Number of memory slots')    
    parser.add_argument('--rm_num_heads', type=int, default=8,     
                        help='Number of heads in relational memory')    
    parser.add_argument('--rm_d_model', type=int, default=512,     
                        help='Dimension of relational memory')    
        
    # Visual extractor parameters    
    parser.add_argument('--visual_extractor', type=str, default='resnet101',     
                        help='Visual extractor to use')    
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,     
                        help='Whether to use pretrained visual extractor')    
        
    # Additional required parameters for model initialization    
    parser.add_argument('--use_bn', type=int, default=0,     
                        help='whether to use batch normalization')    
    parser.add_argument('--bos_idx', type=int, default=0,     
                        help='the index of <bos>')    
    parser.add_argument('--eos_idx', type=int, default=0,     
                        help='the index of <eos>')    
    parser.add_argument('--pad_idx', type=int, default=0,     
                        help='the index of <pad>')    
        
    return parser.parse_args()  
  
def load_single_image(image_path, transform):  
    """Load and transform a single image"""  
    img = Image.open(image_path).convert("RGB")  
    return transform(img).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]  
  
def get_image_paths(args):  
    """Get a list of image paths from the provided arguments"""  
    image_paths = []  
      
    if args.image_path:  
        # Single image path provided  
        image_paths.append(args.image_path)  
      
    elif args.image_dir:  
        # Directory of images provided  
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '*.jpg')))  
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '*.jpeg')))  
        image_paths.extend(glob.glob(os.path.join(args.image_dir, '*.png')))  
      
    elif args.image_list:  
        # List of image paths in a file  
        with open(args.image_list, 'r') as f:  
            image_paths = [line.strip() for line in f.readlines()]  
      
    else:  
        raise ValueError("Please provide either --image_path, --image_dir, or --image_list")  
      
    return image_paths  
  
def prepare_batch(image_paths, args, transform, start_idx, batch_size):  
    """Prepare a batch of images for inference"""  
    end_idx = min(start_idx + batch_size, len(image_paths))  
    batch_paths = image_paths[start_idx:end_idx]  
    batch_images = []  
      
    for path in batch_paths:  
        img = load_single_image(path, transform)  
        batch_images.append(img)  
      
    # Stack images into a batch  
    if args.dataset_name == "iu_xray":  
        # For IU X-ray, each sample needs two identical images  
        batch_tensor = torch.cat([img.repeat(2, 1, 1, 1).unsqueeze(0) for img in batch_images], dim=0)  
    else:  
        # For MIMIC-CXR, just stack the images  
        batch_tensor = torch.cat(batch_images, dim=0)  
      
    return batch_tensor, batch_paths  
  
def main():  
    args = parse_args()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
      
    # Create output directory if it doesn't exist  
    os.makedirs(args.output_dir, exist_ok=True)  
  
    # Set up image transformation  
    transform = transforms.Compose([  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize((0.485, 0.456, 0.406),  
                             (0.229, 0.224, 0.225))  
    ])  
      
    # Get all image paths  
    image_paths = get_image_paths(args)  
    print(f"Found {len(image_paths)} images to process")  
      
    # Load tokenizer  
    print(f"Loading tokenizer from {args.tokenizer_path}")  
    with open(args.tokenizer_path, "rb") as f:  
        tokenizer = pickle.load(f)  
  
    # Create model  
    print("Creating model architecture")  
    model = R2GenModel(args, tokenizer)  
      
    # Load model weights  
    print(f"Loading model weights from {args.model_path}")  
    checkpoint = torch.load(args.model_path, map_location=device)  
    if 'state_dict' in checkpoint:  
        # This is a checkpoint saved by the trainer  
        model.load_state_dict(checkpoint['state_dict'])  
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")  
    else:  
        # This might be just the state dict itself  
        model.load_state_dict(checkpoint)  
        print("Loaded state dictionary directly")  
      
    model.to(device)  
    model.eval()  
      
    # Process images in batches  
    for start_idx in range(0, len(image_paths), args.batch_size):  
        # Prepare batch  
        batch_images, batch_paths = prepare_batch(  
            image_paths, args, transform, start_idx, args.batch_size  
        )  
        batch_images = batch_images.to(device)  
          
        print(f"Processing batch {start_idx//args.batch_size + 1}/{(len(image_paths)-1)//args.batch_size + 1}")  
          
        # Run inference  
        with torch.no_grad():  
            outputs = model(batch_images, mode="sample")  
          
        # Process each output in the batch  
        for i, (output, image_path) in enumerate(zip(outputs, batch_paths)):  
            # Convert token IDs to text  
            report_ids = output.tolist()  
            report_tokens = [tokenizer.idx2token[idx] for idx in report_ids if idx > 0]  
            report = " ".join(report_tokens)  
              
            # Save report to file  
            output_file = os.path.join(  
                args.output_dir,   
                os.path.splitext(os.path.basename(image_path))[0] + "_report.txt"  
            )  
              
            with open(output_file, "w") as f:  
                f.write(report)  
              
            print(f"Report for {os.path.basename(image_path)} saved to {output_file}")  
              
            # Print the first report in each batch as an example  
            if i == 0:  
                print("\nSample Generated Report:")  
                print("-" * 50)  
                print(report)  
                print("-" * 50)  
  
if __name__ == "__main__":  
    main()