#!/usr/bin/env python3
#################################################################
# Script to download test data for PlotData                     #
# Author: Falk Amelung                                          #
# Created: December 2024                                        #
#################################################################

import os
import sys
import argparse
import tarfile
import urllib.request
import urllib.error
from pathlib import Path

# NOTE: Currently downloads from internal server (149.165.154.65)
# Future: Will migrate to Zenodo for public access without authentication
# TO MIGRATE TO ZENODO:
#   1. Update BASE_URL to the Zenodo record URL (e.g., https://zenodo.org/record/XXXXX/files/)

BASE_URL = "http://149.165.154.65/data/circleci"

# Test data files with descriptions
TESTDATA_FILES = {
    'hvGalapagos_mintpy.tar.gz': {
        'description': 'Horizontal and vertical velocity for Galapagos (MintPy)',
        'quick': True  # Used for --quick option
    },
    'hvGalapagos_miaplpy.tar.gz': {
        'description': 'Horizontal and vertical velocity for Galapagos (MiaplPy)',
        'quick': False
    },
    'Fernandina_mintpy.tar.gz': {
        'description': 'Fernandina volcano test dataset (MintPy)',
        'quick': False
    }
}

EXAMPLE = """Examples:
  # Download all test datasets and extract them
  get_plotdata_testdata.py

  # Quick download (only hvGalapagos_mintpy.tar.gz)
  get_plotdata_testdata.py --quick
"""

DESCRIPTION = """
Download test data for PlotData from remote server.

Available datasets:
  Full download (default):
    - hvGalapagos_mintpy.tar.gz     : Galapagos horizontal/vertical velocity (MintPy)
    - hvGalapagos_miaplpy.tar.gz    : Galapagos horizontal/vertical velocity (MiaplPy)
    - Fernandina_mintpy.tar.gz      : Fernandina volcano test dataset (MintPy)
  
  Quick download (--quick):
    - hvGalapagos_mintpy.tar.gz only : Minimal dataset for quick testing

Note: Files will be downloaded to the current directory and automatically extracted.
"""


def create_parser(iargs=None):
    """
    Creates command line argument parser object.

    Args:
        iargs (list): List of command line arguments (default: None)

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument('--quick',
                        action='store_true',
                        default=False,
                        help='Download only the minimal dataset (hvGalapagos_mintpy.tar.gz) for quick testing')

    inps = parser.parse_args(iargs)

    return inps


def download_file(url, destination, filename):
    """
    Download a file from URL with progress reporting.

    Args:
        url (str): Full URL of the file to download
        destination (str): Directory to save the file
        filename (str): Name of the file

    Returns:
        bool: True if download successful, False otherwise
    """
    output_path = os.path.join(destination, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print(f"Skipping {filename}")
            return True

    try:
        print(f"\nDownloading: {filename}")
        print(f"From: {url}")
        
        # Open URL and get file size
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            
            if file_size > 0:
                file_size_mb = file_size / (1024 * 1024)
                print(f"Size: {file_size_mb:.2f} MB")
            
            # Download with progress
            downloaded = 0
            block_size = 8192
            
            with open(output_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    # Show progress
                    if file_size > 0:
                        percent = (downloaded / file_size) * 100
                        downloaded_mb = downloaded / (1024 * 1024)
                        print(f"\rProgress: {percent:.1f}% ({downloaded_mb:.2f} MB / {file_size_mb:.2f} MB)", 
                              end='', flush=True)
        
        print()  # New line after progress
        print(f"✓ Successfully downloaded: {output_path}")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        print(f"  Failed to download: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        print(f"  Failed to download: {url}")
        print("  Check your network connection and that the server is accessible.")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error downloading {filename}: {str(e)}")
        return False


def extract_tarfile(filepath, destination):
    """
    Extract a tar.gz file.

    Args:
        filepath (str): Path to the tar.gz file
        destination (str): Directory to extract to

    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        print(f"\nExtracting: {os.path.basename(filepath)}")
        
        with tarfile.open(filepath, 'r:gz') as tar:
            # Get list of members to show progress
            members = tar.getmembers()
            total_files = len(members)
            
            print(f"Extracting {total_files} files...")
            
            # Extract all files
            tar.extractall(path=destination)
        
        print(f"✓ Successfully extracted to: {destination}")
        return True
        
    except tarfile.TarError as e:
        print(f"✗ Error extracting {filepath}: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error extracting {filepath}: {str(e)}")
        return False


def main(iargs=None):
    """
    Main function to download PlotData test data.

    Args:
        iargs (list): List of command line arguments (default: None)
    """
    inps = create_parser(iargs)
    
    # Use current directory
    outdir = os.getcwd()
    print(f"Output directory: {outdir}")
    
    # Determine which files to download
    if inps.quick:
        files_to_download = {k: v for k, v in TESTDATA_FILES.items() if v['quick']}
        print("\n=== Quick mode: Downloading minimal dataset ===")
    else:
        files_to_download = TESTDATA_FILES
        print("\n=== Downloading all test datasets ===")
    
    # Download files
    successful_downloads = []
    failed_downloads = []
    
    for filename, info in files_to_download.items():
        url = f"{BASE_URL}/{filename}"
        
        if download_file(url, outdir, filename):
            successful_downloads.append(filename)
        else:
            failed_downloads.append(filename)
    
    # Extract files
    if successful_downloads:
        print("\n=== Extracting downloaded files ===")
        
        for filename in successful_downloads:
            filepath = os.path.join(outdir, filename)
            
            if extract_tarfile(filepath, outdir):
                # Optionally remove tar.gz after successful extraction
                # Uncomment the following lines if you want to auto-delete archives
                # print(f"Removing archive: {filename}")
                # os.remove(filepath)
                pass
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful downloads: {len(successful_downloads)}/{len(files_to_download)}")
    
    if successful_downloads:
        print("\n✓ Downloaded files:")
        for filename in successful_downloads:
            print(f"  - {filename}")
    
    if failed_downloads:
        print("\n✗ Failed downloads:")
        for filename in failed_downloads:
            print(f"  - {filename}")
        print("\nPlease check your network connection and try again.")
        sys.exit(1)
    
    if successful_downloads:
        print(f"\nFiles extracted to: {outdir}")
    
    print("\n✓ All downloads completed successfully!")
    print("\nYou can now use these test datasets with PlotData.")


if __name__ == "__main__":
    main()

