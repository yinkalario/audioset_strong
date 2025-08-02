#!/bin/bash

# AudioSet Label Distribution Analysis Script
# This script runs both strong and weak label distribution analyses

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
if [ ! -f "src/analyze_strong_label_distribution.py" ]; then
    print_error "Please run this script from the audioset_strong root directory"
    exit 1
fi

# Check if required files exist
if [ ! -f "meta/audioset_train_strong.tsv" ]; then
    print_error "Strong labeling file not found: meta/audioset_train_strong.tsv"
    exit 1
fi

if [ ! -f "meta/unbalanced_train_segments.csv" ]; then
    print_error "Weak labeling file not found: meta/unbalanced_train_segments.csv"
    exit 1
fi

if [ ! -f "meta/mid_to_display_name.tsv" ]; then
    print_warning "Label mapping file not found: meta/mid_to_display_name.tsv"
    print_warning "Analysis will proceed without display names"
fi

print_step "Starting AudioSet Label Distribution Analysis"

# Create output directory
mkdir -p out

# Step 1: Analyze strong label distribution
print_step "Step 1: Analyzing Strong Label Distribution (by duration)"
python3 src/analyze_strong_label_distribution.py \
    --train-file meta/audioset_train_strong.tsv \
    --mid-to-display meta/mid_to_display_name.tsv

print_success "Strong label analysis completed"

# Step 2: Analyze weak label distribution  
print_step "Step 2: Analyzing Weak Label Distribution (by occurrence count)"
python3 src/analyze_weak_label_distribution.py \
    --train-file meta/unbalanced_train_segments.csv \
    --mid-to-display meta/mid_to_display_name.tsv \
    --output-dir out

print_success "Weak label analysis completed"

# Step 3: Rename files for consistency
print_step "Step 3: Renaming Output Files for Consistency"

# Rename strong label analysis files
if [ -f "out/label_distribution_analysis.png" ]; then
    mv "out/label_distribution_analysis.png" "out/strong_label_distribution_analysis.png"
    print_success "Renamed to strong_label_distribution_analysis.png"
fi

if [ -f "out/top_labels_detailed.png" ]; then
    mv "out/top_labels_detailed.png" "out/top_strong_labels_detailed.png"
    print_success "Renamed to top_strong_labels_detailed.png"
fi

if [ -f "out/label_distribution_stats.csv" ]; then
    mv "out/label_distribution_stats.csv" "out/strong_label_distribution_stats.csv"
    print_success "Renamed to strong_label_distribution_stats.csv"
fi

# Rename weak label analysis files (already have good names, but ensure consistency)
if [ -f "out/weak_label_distribution_analysis.png" ]; then
    print_success "Weak label analysis files already have consistent naming"
fi

print_step "Analysis Complete!"
print_success "All label distribution analyses completed successfully"

# Display summary
print_step "Generated Files Summary"
echo "Strong Label Analysis (by duration):"
echo "├── out/strong_label_distribution_analysis.png  # 6-panel comprehensive analysis"
echo "├── out/top_strong_labels_detailed.png          # Top 50 labels by duration"
echo "└── out/strong_label_distribution_stats.csv     # Complete statistics"
echo ""
echo "Weak Label Analysis (by occurrence count):"
echo "├── out/weak_label_distribution_analysis.png    # 6-panel comprehensive analysis"
echo "├── out/top_weak_labels_detailed.png            # Top 50 labels by occurrence"
echo "└── out/weak_label_distribution_stats.csv       # Complete statistics"
echo ""

print_success "Ready for analysis review!"
print_warning "Check the out/ directory for all generated plots and statistics"
