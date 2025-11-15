import os
import tarfile
import urllib.request
import sys

def download_with_progress(url: str, filename: str):
    """Download file with progress bar"""
    def download_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% [{mb_downloaded:.1f}/{mb_total:.1f} MB]')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, reporthook=download_hook)
        print()  # New line after download completes
    except Exception as e:
        print(f"\nError downloading file: {e}")
        raise

def download_voc(data_dir: str, year: str = "2012"):
    """Download and extract PASCAL VOC dataset"""
    if year not in ["2007", "2012"]:
        raise ValueError("Year must be '2007' or '2012'")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for different years
    urls = {
        "2007": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "2012": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    }
    
    url = urls[year]
    filename = os.path.join(data_dir, f"VOC{year}.tar")
    
    # Download if not already present
    if not os.path.exists(filename):
        print(f"Downloading VOC{year} dataset from {url}")
        download_with_progress(url, filename)
    
    # Extract if not already extracted
    if not os.path.exists(os.path.join(data_dir, f"VOC{year}")):
        print(f"Extracting VOC{year} dataset...")
        with tarfile.open(filename) as tar:
            tar.extractall(data_dir)
    
    print("Dataset ready!")

if __name__ == "__main__":
    # Example usage
    download_voc("./VOCdevkit", "2012")
