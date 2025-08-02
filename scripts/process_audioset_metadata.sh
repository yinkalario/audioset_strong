#!/bin/bash

# AudioSet Strong Labeling Metadata Processing Pipeline
# This script runs all the metadata processing steps in the correct sequence

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "src/generate_raw_target_meta.py" ]; then
    print_error "Please run this script from the audioset_strong root directory"
    exit 1
fi

# Check if meta directory exists with required files
if [ ! -d "meta" ] || [ ! -f "meta/audioset_train_strong.tsv" ]; then
    print_error "Meta directory or required AudioSet files not found"
    print_warning "Please ensure you have downloaded the AudioSet strong labeling files to meta/"
    exit 1
fi

print_step "Starting AudioSet Strong Labeling Metadata Processing Pipeline"

# Sound types to process
SOUND_TYPES=("baby_cry" "gun" "snore")

# Step 1: Generate raw positive metadata for each sound type
print_step "Step 1: Generating Raw Positive Metadata"

for sound_type in "${SOUND_TYPES[@]}"; do
    print_step "Processing $sound_type positive labels"
    
    # Update the target configuration in the script
    case $sound_type in
        "baby_cry")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/t\/dd00002'\'']  # Baby cry, infant cry/' src/generate_raw_target_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''baby_cry'\''/' src/generate_raw_target_meta.py
            ;;
        "gun")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/032s66'\'', '\''\/m\/04zjc'\'', '\''\/m\/073cg4'\'']  # Gunshot\/gunfire, Machine gun, Cap gun/' src/generate_raw_target_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''gun'\''/' src/generate_raw_target_meta.py
            ;;
        "snore")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/01d3sd'\'', '\''\/m\/07q0yl5'\'']  # Snoring, Snort/' src/generate_raw_target_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''snore'\''/' src/generate_raw_target_meta.py
            ;;
    esac
    
    python3 src/generate_raw_target_meta.py --input-dir meta --output-dir meta
    print_success "Generated raw positive metadata for $sound_type"
done

# Step 2: Generate raw negative strong metadata for each sound type
print_step "Step 2: Generating Raw Negative Strong Metadata"

for sound_type in "${SOUND_TYPES[@]}"; do
    print_step "Processing $sound_type negative strong labels"

    # Update the target configuration in the script
    case $sound_type in
        "baby_cry")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/t\/dd00002'\'']  # Baby cry, infant cry/' src/generate_raw_neg_strong_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''baby_cry'\''/' src/generate_raw_neg_strong_meta.py
            ;;
        "gun")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/032s66'\'', '\''\/m\/04zjc'\'', '\''\/m\/073cg4'\'']  # Gunshot\/gunfire, Machine gun, Cap gun/' src/generate_raw_neg_strong_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''gun'\''/' src/generate_raw_neg_strong_meta.py
            ;;
        "snore")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/01d3sd'\'', '\''\/m\/07q0yl5'\'']  # Snoring, Snort/' src/generate_raw_neg_strong_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''snore'\''/' src/generate_raw_neg_strong_meta.py
            ;;
    esac

    python3 src/generate_raw_neg_strong_meta.py --input-dir meta --output-dir meta
    print_success "Generated raw negative strong metadata for $sound_type"
done

# Step 3: Generate raw negative weak metadata for each sound type
print_step "Step 3: Generating Raw Negative Weak Metadata"

for sound_type in "${SOUND_TYPES[@]}"; do
    print_step "Processing $sound_type negative weak labels"

    # Update the target configuration in the script
    case $sound_type in
        "baby_cry")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/t\/dd00002'\'']  # Baby cry, infant cry/' src/generate_raw_neg_weak_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''baby_cry'\''/' src/generate_raw_neg_weak_meta.py
            ;;
        "gun")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/032s66'\'', '\''\/m\/04zjc'\'', '\''\/m\/073cg4'\'']  # Gunshot\/gunfire, Machine gun, Cap gun/' src/generate_raw_neg_weak_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''gun'\''/' src/generate_raw_neg_weak_meta.py
            ;;
        "snore")
            sed -i.bak 's/^target_labels = .*/target_labels = ['\''\/m\/01d3sd'\'', '\''\/m\/07q0yl5'\'']  # Snoring, Snort/' src/generate_raw_neg_weak_meta.py
            sed -i.bak 's/^target_name = .*/target_name = '\''snore'\''/' src/generate_raw_neg_weak_meta.py
            ;;
    esac

    python3 src/generate_raw_neg_weak_meta.py --input-dir meta --output-dir meta
    print_success "Generated raw negative weak metadata for $sound_type"
done

# Step 4: Generate segmented positive metadata
print_step "Step 4: Generating Segmented Positive Metadata"
python3 src/generate_seg_target_meta.py
print_success "Generated segmented positive metadata for all sound types"

# Step 5: Generate segmented negative metadata
print_step "Step 5: Generating Segmented Negative Metadata"
python3 src/generate_seg_neg_strong_meta.py
print_success "Generated segmented negative metadata for all sound types"

# Clean up backup files
rm -f src/*.bak

print_step "Pipeline Complete!"
print_success "All metadata processing steps completed successfully"

# Display summary
print_step "Summary"
echo "Generated metadata for sound types: ${SOUND_TYPES[*]}"
echo "Directory structure:"
echo "meta/"
for sound_type in "${SOUND_TYPES[@]}"; do
    echo "├── $sound_type/"
    echo "│   ├── raw/"
    echo "│   │   ├── pos/          # Raw positive labels"
    echo "│   │   ├── neg_strong/   # Raw negative strong labels"
    echo "│   │   └── neg_weak/     # Raw negative weak labels (10-second segments)"
    echo "│   └── seg1s/"
    echo "│       ├── pos/          # 1-second segmented positive labels"
    echo "│       └── neg_strong/   # 1-second segmented negative labels"
done

print_success "Ready for dataloader implementation!"
