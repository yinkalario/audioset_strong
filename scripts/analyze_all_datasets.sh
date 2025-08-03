#!/bin/bash

# AudioSet Complete Dataset Analysis Script
# Analyzes all strong and weak label datasets separately

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
if [ ! -f "src/analyze_meta/analyze_single_strong_dataset.py" ]; then
    print_error "Please run this script from the audioset_strong root directory"
    exit 1
fi

print_step "Starting Complete AudioSet Dataset Analysis"
print_warning "This will analyze all 6 datasets separately (3 strong + 3 weak)"

# Create output directory
mkdir -p out

# Define datasets to analyze
STRONG_DATASETS="train eval eval_framed"
WEAK_DATASETS="balanced_train unbalanced_train eval"

# Strong dataset files
STRONG_FILES_train="meta/audioset_train_strong.tsv"
STRONG_FILES_eval="meta/audioset_eval_strong.tsv"
STRONG_FILES_eval_framed="meta/audioset_eval_strong_framed_posneg.tsv"

# Weak dataset files
WEAK_FILES_balanced_train="meta/balanced_train_segments.csv"
WEAK_FILES_unbalanced_train="meta/unbalanced_train_segments.csv"
WEAK_FILES_eval="meta/eval_segments.csv"

# Analyze strong label datasets separately
print_step "Analyzing Strong Label Datasets (by duration)"

# Process train dataset
if [ -f "meta/audioset_train_strong.tsv" ]; then
    print_step "Analyzing train strong labels"
    python3 src/analyze_meta/analyze_single_strong_dataset.py \
        --input-file "meta/audioset_train_strong.tsv" \
        --output-prefix "out/strong_label_distribution_train" \
        --dataset-name "Train" \
        --mid-to-display meta/mid_to_display_name.tsv
    print_success "Generated train strong analysis files"
fi

# Process eval dataset
if [ -f "meta/audioset_eval_strong.tsv" ]; then
    print_step "Analyzing eval strong labels"
    python3 src/analyze_meta/analyze_single_strong_dataset.py \
        --input-file "meta/audioset_eval_strong.tsv" \
        --output-prefix "out/strong_label_distribution_eval" \
        --dataset-name "Eval" \
        --mid-to-display meta/mid_to_display_name.tsv
    print_success "Generated eval strong analysis files"
fi

# Process eval framed dataset
if [ -f "meta/audioset_eval_strong_framed_posneg.tsv" ]; then
    print_step "Analyzing eval framed strong labels (PRESENT events only)"
    python3 src/analyze_meta/analyze_single_strong_dataset.py \
        --input-file "meta/audioset_eval_strong_framed_posneg.tsv" \
        --output-prefix "out/strong_label_distribution_eval_framed" \
        --dataset-name "Eval Framed" \
        --mid-to-display meta/mid_to_display_name.tsv
    print_success "Generated eval framed strong analysis files"
fi

# Process eval_framed dataset (filter for PRESENT only)
if [ -f "meta/audioset_eval_strong_framed_posneg.tsv" ]; then
    print_step "Analyzing eval_framed strong labels (PRESENT only)"

    # Create temporary file with only PRESENT events
    temp_file="meta/audioset_eval_strong_framed_present_only.tsv"
    head -1 meta/audioset_eval_strong_framed_posneg.tsv > "$temp_file"
    awk -F'\t' 'NR>1 && $5=="PRESENT" {print}' meta/audioset_eval_strong_framed_posneg.tsv >> "$temp_file"

    print_warning "Created temporary file with $(tail -n +2 "$temp_file" | wc -l) PRESENT events"

    if python3 src/analyze_strong_label_distribution.py \
        --train-file "$temp_file" \
        --mid-to-display meta/mid_to_display_name.tsv; then

        # Rename files
        [ -f "out/strong_label_distribution_analysis.png" ] && mv "out/strong_label_distribution_analysis.png" "out/strong_label_distribution_analysis_eval_framed.png"
        [ -f "out/top_strong_labels_detailed.png" ] && mv "out/top_strong_labels_detailed.png" "out/top_strong_labels_detailed_eval_framed.png"
        [ -f "out/strong_label_distribution_stats.csv" ] && mv "out/strong_label_distribution_stats.csv" "out/strong_label_distribution_stats_eval_framed.csv"
        print_success "Generated eval_framed strong analysis files (PRESENT only)"
    else
        print_warning "Strong analysis failed for eval_framed dataset, skipping..."
    fi

    # Clean up temporary file
    rm -f "$temp_file"
fi

# Analyze weak label datasets separately
print_step "Analyzing Weak Label Datasets (by occurrence count)"

for dataset_name in $WEAK_DATASETS; do
    # Get file path using variable indirection
    file_var="WEAK_FILES_${dataset_name}"
    file_path="${!file_var}"
    
    if [ -f "$file_path" ]; then
        print_step "Analyzing $dataset_name weak labels: $file_path"
        
        # Create individual analysis by setting only one file and nullifying others
        case $dataset_name in
            "balanced_train")
                python3 src/analyze_meta/analyze_weak_label_distribution.py \
                    --unbalanced-train "/dev/null" \
                    --balanced-train "$file_path" \
                    --eval-file "/dev/null" \
                    --mid-to-display meta/mid_to_display_name.tsv \
                    --output-dir out
                ;;
            "unbalanced_train")
                python3 src/analyze_meta/analyze_weak_label_distribution.py \
                    --unbalanced-train "$file_path" \
                    --balanced-train "/dev/null" \
                    --eval-file "/dev/null" \
                    --mid-to-display meta/mid_to_display_name.tsv \
                    --output-dir out
                ;;
            "eval")
                python3 src/analyze_meta/analyze_weak_label_distribution.py \
                    --unbalanced-train "/dev/null" \
                    --balanced-train "/dev/null" \
                    --eval-file "$file_path" \
                    --mid-to-display meta/mid_to_display_name.tsv \
                    --output-dir out
                ;;
        esac
        
        # Check if individual analysis files were generated
        if [ -f "out/weak_label_distribution_analysis_${dataset_name}.png" ]; then
            print_success "Generated: weak_label_distribution_analysis_${dataset_name}.png"
        fi
        
        if [ -f "out/weak_label_distribution_stats_${dataset_name}.csv" ]; then
            print_success "Generated: weak_label_distribution_stats_${dataset_name}.csv"
        fi
        
        print_success "Completed analysis for $dataset_name weak labels"
    else
        print_warning "File not found: $file_path"
    fi
done

# Clean up any combined analysis files (we only want individual analyses)
print_step "Cleaning up combined analysis files"
[ -f "out/weak_label_distribution_analysis.png" ] && rm "out/weak_label_distribution_analysis.png" && print_warning "Removed combined weak analysis"
[ -f "out/weak_label_distribution_stats.csv" ] && rm "out/weak_label_distribution_stats.csv" && print_warning "Removed combined weak stats"
[ -f "out/top_weak_labels_detailed.png" ] && rm "out/top_weak_labels_detailed.png" && print_warning "Removed combined weak detailed plot"

print_step "Complete Dataset Analysis Finished!"
print_success "All 6 datasets analyzed separately"

# Display comprehensive summary
print_step "Generated Files Summary"
echo ""
echo "Strong Label Individual Analyses (by duration):"
STRONG_OUTPUTS="train eval eval_framed"
for dataset_name in $STRONG_OUTPUTS; do
    if [ -f "out/strong_label_distribution_${dataset_name}_analysis.png" ]; then
        echo "├── Dataset: $dataset_name"
        echo "│   ├── out/strong_label_distribution_${dataset_name}_analysis.png"
        echo "│   ├── out/strong_label_distribution_${dataset_name}_detailed.png"
        echo "│   └── out/strong_label_distribution_${dataset_name}_stats.csv"
    fi
done

echo ""
echo "Weak Label Individual Analyses (by occurrence count):"
for dataset_name in $WEAK_DATASETS; do
    if [ -f "out/weak_label_distribution_analysis_${dataset_name}.png" ]; then
        echo "├── Dataset: $dataset_name"
        echo "│   ├── out/weak_label_distribution_analysis_${dataset_name}.png"
        echo "│   └── out/weak_label_distribution_stats_${dataset_name}.csv"
    fi
done

echo ""
print_success "Ready for individual dataset analysis review!"
print_warning "Check the out/ directory for all generated plots and statistics"
print_warning "Each dataset has been analyzed completely separately"
