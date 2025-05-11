import os
from datetime import datetime
import argparse
import torch
from DataLoaders.utils import col_index

def get_config(sysv):
    parser = argparse.ArgumentParser(description='Training variables.')

    parser.add_argument('--model_basename', default='baseline')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank')
    parser.add_argument('--exp_name', default=datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    parser.add_argument('--fold', default="1")
    parser.add_argument('--emo', type=int, default=0, help='Index of emotion in loo experiments')

    parser.add_argument('--pretrained', default=None, help="Path to pretrained weights")
    parser.add_argument('--fromcheckpoint', default=None, help="Path to pretrained weights")
    parser.add_argument('--finetune', action='store_true', help="finetune CLIP")
    parser.set_defaults(finetune=False)
    parser.add_argument('--text', action='store_true', help="finetune CLIP text encoder")
    parser.set_defaults(text=False)
    parser.add_argument('--visual', action='store_true', help="finetune CLIP visual encoder")
    parser.set_defaults(visual=False)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision (AMP) for training')
                        
    parser.add_argument('--resample', action='store_true', help="Resample minority classes")
    parser.set_defaults(resample=False)

    # Set DDP to False by default to avoid distributed training issues
    parser.add_argument('--DDP', action='store_true', default=False, help="Use Distributed Data Parallel")
    
    parser.add_argument('--log_dir', default='logs')
    # Reduce batch size to 1 to avoid memory issues
    parser.add_argument('--batch_size', type=int, default=1, 
                    help='Batch size per GPU. Use gradient accumulation for effective larger batches')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help='Number of steps to accumulate gradients for effective batch size')
    # Reduce number of workers to 1 to avoid memory issues
    parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of data loading workers')
    parser.add_argument('--enable_memory_opt', action='store_true', default=True,
                    help='Enable memory optimizations')
    
    # Add memory efficient options
    parser.add_argument('--max_frames', type=int, default=100,
                    help='Maximum number of frames to load per video')
    parser.add_argument('--frame_size', type=int, default=112,
                    help='Size to resize frames (112 is more memory efficient than 224)')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                    help='Pin memory in DataLoader (disable for less memory usage)')
                    
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--optim', type=str, default='SGD')
   
    parser.add_argument('--lr', type=float, default=1e-4,  # 0.0001
                    help='Base learning rate')
    parser.add_argument('--wd', type=float, default=0.01,   
                    help='Weight decay for regularization')

    parser.add_argument('--dataset_root', help="/home/fyp/Downloads/EmotiW_2018/Train_AFEW")
    # Add TextAnnotatedDataset as a possible dataset
    parser.add_argument('--dataset_name', default='TextAnnotatedDataset')
  
    # Reduce input shape to save memory
    parser.add_argument('--input_shape', nargs='+', default=112)
    parser.add_argument('--downsample', type=int, default=2)
    # Reduce clip length to save memory
    parser.add_argument('--clip_len', type=int, default=16)

    parser.add_argument('--visual_unfreeze', action='store_true', default=True, 
                    help='Unfreeze visual CLIP parameters')
    parser.add_argument('--text_unfreeze', action='store_true', default=False, 
                    help='Unfreeze text CLIP parameters')
    
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes in the dataset')

    parser.add_argument('--debug', action='store_true', default=True,
                    help='Enable debug mode with more verbose output')

    args, _ = parser.parse_known_args(sysv)
    
    # Force some settings for memory efficiency
    args.DDP = False  # Disable distributed training
    
    return args
