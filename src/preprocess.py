"""
Main preprocessing script for the HybridDeepfakeDetector project.
Processes videos in the data directory to extract frames and transcribe audio.
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import config
from utils.data_utils import VideoProcessor, AudioProcessor, is_video_processed, get_video_files, create_processing_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_video(video_path: str, output_base_dir: str, 
                        video_processor: VideoProcessor, audio_processor: AudioProcessor,
                        skip_existing: bool = True) -> dict:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        output_base_dir: Base output directory
        video_processor: VideoProcessor instance
        audio_processor: AudioProcessor instance
        skip_existing: Whether to skip already processed videos
        
    Returns:
        Dictionary with processing results
    """
    video_name = Path(video_path).stem
    
    # Check if already processed
    if skip_existing and is_video_processed(video_path, output_base_dir):
        logger.info(f"Skipping {video_name} - already processed")
        return {
            "video_path": video_path,
            "video_name": video_name,
            "success": True,
            "skipped": True,
            "face_count": 0,
            "transcript_length": 0
        }
    
    logger.info(f"Processing video: {video_name}")
    
    try:
        # Create output directories
        frames_dir = os.path.join(output_base_dir, f"{video_name}_frames")
        
        # Process video frames
        logger.info(f"Extracting frames and faces from {video_name}")
        face_count = video_processor.process_video_frames(
            video_path=video_path,
            output_dir=frames_dir,
            fps=config.FRAMES_PER_SECOND,
            target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        
        # Process audio transcription
        logger.info(f"Transcribing audio from {video_name}")
        transcription_data = audio_processor.transcribe_video(video_path)
        
        # Save transcription
        transcript_path = os.path.join(output_base_dir, f"{video_name}_transcript.txt")
        audio_processor.save_transcription(transcription_data, transcript_path)
        
        result = {
            "video_path": video_path,
            "video_name": video_name,
            "success": True,
            "skipped": False,
            "face_count": face_count,
            "transcript_length": len(transcription_data["text"]),
            "frames_dir": frames_dir,
            "transcript_path": transcript_path
        }
        
        logger.info(f"Successfully processed {video_name}: {face_count} faces, {len(transcription_data['text'])} chars")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {video_name}: {str(e)}")
        return {
            "video_path": video_path,
            "video_name": video_name,
            "success": False,
            "skipped": False,
            "error": str(e),
            "face_count": 0,
            "transcript_length": 0
        }

def preprocess_dataset(data_dir: str, output_dir: str, skip_existing: bool = True):
    """
    Preprocess entire dataset.
    
    Args:
        data_dir: Directory containing train/test folders with real/fake subfolders
        output_dir: Output directory for processed data
        skip_existing: Whether to skip already processed videos
    """
    logger.info("Starting dataset preprocessing")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    logger.info("Initializing video and audio processors")
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor(model_name=config.WHISPER_MODEL)
    
    # Process each subset (train/test) and class (real/fake)
    subsets = ['train', 'test']
    classes = ['real', 'fake']
    
    all_results = []
    
    for subset in subsets:
        for class_name in classes:
            subset_dir = os.path.join(data_dir, subset, class_name)
            
            if not os.path.exists(subset_dir):
                logger.warning(f"Directory not found: {subset_dir}")
                continue
            
            logger.info(f"Processing {subset}/{class_name}")
            
            # Get video files
            video_files = get_video_files(subset_dir, config.VIDEO_EXTENSIONS)
            
            if not video_files:
                logger.warning(f"No video files found in {subset_dir}")
                continue
            
            # LIMIT TO FIRST 500 VIDEOS FOR TESTING
            video_files = video_files[:500]
            logger.info(f"Found {len(video_files)} videos in {subset}/{class_name} (limited to first 500)")
            
            # Create output directory for this subset/class
            output_subset_dir = os.path.join(output_dir, subset, class_name)
            Path(output_subset_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each video with progress bar
            for video_path in tqdm(video_files, desc=f"Processing {subset}/{class_name}"):
                result = process_single_video(
                    video_path=video_path,
                    output_base_dir=output_subset_dir,
                    video_processor=video_processor,
                    audio_processor=audio_processor,
                    skip_existing=skip_existing
                )
                
                # Add subset and class information
                result['subset'] = subset
                result['class'] = class_name
                all_results.append(result)
    
    # Create processing summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    create_processing_summary(all_results, summary_path)
    
    # Print final statistics
    total_videos = len(all_results)
    successful = sum(1 for r in all_results if r['success'])
    skipped = sum(1 for r in all_results if r.get('skipped', False))
    failed = total_videos - successful
    total_faces = sum(r.get('face_count', 0) for r in all_results)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total faces extracted: {total_faces}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess videos for deepfake detection")
    
    # TODO: Modify these default paths if needed
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=str(config.DATA_DIR),
        help="Directory containing train/test folders with real/fake subfolders"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(config.DATA_DIR / "processed"),
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--skip_existing", 
        action="store_true", 
        default=True,
        help="Skip already processed videos"
    )
    
    parser.add_argument(
        "--no_skip", 
        action="store_true", 
        help="Process all videos even if already processed"
    )
    
    args = parser.parse_args()
    
    # Handle skip_existing logic
    skip_existing = args.skip_existing and not args.no_skip
    
    # Print configuration
    config.print_config()
    
    # Start preprocessing
    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_existing=skip_existing
    )

if __name__ == "__main__":
    main()
